# Siphon

Siphon is a from-scratch LLM inference engine in raw C for Linux. Its goal is to
run existing GGUF models (up to ~36B parameters) with the best possible speed and
zero quality loss on weak hardware (think: 6-core CPU + 4GB-VRAM GPU + 16GB RAM),
splitting work across GPU (Vulkan compute only — no CUDA), CPU, and disk, and
exposing it all to agent harnesses (opencode, Claude Code) through
OpenAI/Anthropic-compatible endpoints.

Hard rules the project holds itself to:

- Raw C (C17), Linux only. No C++, no Python in the build or runtime.
- Near-zero dependencies: libc, libm, pthreads. The Vulkan loader will be
  `dlopen`ed at runtime so the binary still runs with no GPU.
- Everything hand-written: GGUF parser, tokenizers, HTTP/SSE server, chat
  templates, JSON. Shaders are GLSL compiled to SPIR-V at development time and
  committed as C arrays; the build needs only `cc` and `make`.
- Nothing lossy by default. Any lossy option must be opt-in and clearly labeled.
- No fake benchmarks. Numbers below are real measurements.

## Status

| Milestone | State |
|---|---|
| 1. GGUF parser + tokenizers + `siphon info` | **done** (see below) |
| 2. CPU-only inference (1–3B llama-family) | **done** (see below) |
| 3. Vulkan backend via dlopen loader | not started |
| 4. Hybrid GPU/CPU/disk scheduler | not started |
| 5. Resource modes + safety floor + watchdog | not started |
| 6. Paged KV cache + persistent prefix cache | not started |
| 7. OpenAI/Anthropic server + tool calling | not started |
| 8. MoE, speculative decoding, 36B tuning | not started |

### Milestone 1 — what works

- **GGUF v2/v3 parser** (little-endian): mmap-based and zero-copy, all metadata
  value types, tensor infos with exact quantized-size geometry for every ggml
  tensor type (classic quants, K-quants, IQ/TQ/MXFP4/NVFP4 — geometry only;
  dequantization arrives with milestone 2). Alignment handling, bounds checking,
  and clean rejection of malformed files.
- **Tokenizers**, bit-for-bit identical to llama.cpp (used as a dev-time
  reference only, never linked):
  - SentencePiece family ("llama"): score-priority BPE merges, `▁` space
    escaping, dummy prefix, `<0xXX>` byte fallback.
  - Byte-level BPE family ("gpt2") with the pretokenizer flavors real models
    use (`gpt-2`, `qwen2` for Qwen2/2.5/3, `llama3`), GPT-2 byte-level
    encoding, rank-priority merges, special-token partitioning, and the
    reference's EOG promotion heuristics (e.g. Qwen's plain `</s>` token).
  - Detokenization with the reference's exact piece rules (byte-level decode,
    control-token suppression, leading-space handling, clean_spaces passes).
- **`siphon info`**: full model report (metadata, tokenizer summary, per-type
  tensor statistics, bits/weight). **`siphon tokenize`**: verification tool
  with ids/pieces/roundtrip output and a machine-readable `--hex` mode.

### Verification (real numbers, this sandbox: 2 cores, gcc 14.2, -O2)

- Unit tests: 172 checks green under ASan+UBSan (parser roundtrips, malformed
  input, synthetic SPM/BPE vocabularies, unicode tables, byte-level mapping).
- **Reference-diffed fixtures**: 46-case battery (English, contractions,
  CJK, emoji+ZWJ, Cyrillic, invalid UTF-8, NUL bytes, control chars, special
  tokens, long runs) × 3 tokenize modes × 2 models = **276 cases, 0
  differences** against llama.cpp's tokenizer, plus byte-identical
  detokenization. Fixtures are committed under `tests/fixtures/`.
- Tokenizer throughput on the full battery repeated 200x (~660 KB text):
  TinyLlama-1.1B (SPM): **~2.4M tokens/s**; Qwen2.5-0.5B (BPE): **~0.6M
  tokens/s**; `siphon info` metadata parse: **~0.9 ms** (mmap is lazy, so this
  stays flat for 100GB files).
- Models used: TinyLlama-1.1B-Chat-v1.0 Q4_K_M (SentencePiece) and
  Qwen2.5-0.5B-Instruct Q4_K_M (byte-level BPE, `pre=qwen2`).

### Known limitations (milestone 1)

- GGUF v1 files are rejected (ancient, structurally different). Big-endian
  files are rejected (target hardware is x86).
- Tokenizer families beyond SPM/BPE (WPM, UGM, RWKV, PLaMo) are not
  implemented — none of the target models (Llama, Mistral, Qwen2/3) use them.
- Pretokenizer flavors other than gpt-2/qwen2/llama3 (deepseek, falcon,
  chameleon, ...) fall back to gpt-2 rules with a warning; tokenization for
  such models may differ from their original tokenizers.
- `tokenizer.ggml.pre` missing → 'default' pretokenizer without the reference's
  extra number/punctuation refinement regexes (warned; not needed for target
  models).
- LSTRIP/RSTRIP special-token attributes (phi-3/jina models) are ignored.
- `general.file_type` is printed but not validated against tensor types.

### Milestone 2 — what works

- **Numerical kernels** for every weight format in the target models:
  F16, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, plus the runtime
  Q8_K/Q8_0 activation quantizers. Each has a portable scalar reference and
  an AVX2+FMA+F16C implementation compiled with target attributes (one
  binary runs anywhere, CPUID dispatch at startup, `SI_SCALAR=1` forces the
  scalar path). The AVX2 kernels are transcribed structure-for-structure
  from the reference's x86 kernels — lane grouping, maddubs/madd pairing,
  scale shuffles, FMA accumulation and horizontal-sum order included — and
  were verified **bit-exact (0 ulp)** against the reference's dispatched
  kernels with a dev-time differ (`kdiff`): all dot products, the f16 dot
  including its scalar tail, and byte-identical q8_0/q8_K activation
  quantization.
- **Elementwise numerics** mirror the reference exactly: its exp
  polynomial transcribed to scalar `fmaf` form (lane-identical to the
  vector code — proven by test), silu, softmax (8-wide exp groups into a
  double total, then the reciprocal multiply), rmsnorm (double-accumulated
  sum of f32 squares). The f16 conversions are software round-to-nearest-
  even and match F16C hardware over 19.9M random floats plus all 65536
  roundtrips.
- **Persistent thread pool**: workers spin briefly then park on a
  condition variable; tasks receive `(lane, n_lanes)` and partition work
  deterministically. Every output row is computed start-to-finish inside
  one lane (and the attention keeps per-lane scratch), which the tests
  prove makes results **independent of the thread count**.
- **Backend abstraction**: tensors carry location tags and a backend
  vtable (`gemv`, `row_to_f32`); the model assigns each layer's weights to
  a backend. Milestone 3 adds the Vulkan backend without changing any
  call site. The CPU `gemv` converts the activation row once into the
  weight type's dot-pair format — the same work-buffer semantics as the
  reference.
- **Model loader** for the llama and qwen2 graphs: hyperparameters,
  zero-copy weight views, GQA, tied or separate output heads, q/k/v
  biases (Qwen), F32-materialized norms.
- **Engine**: token-serial decode; K cache position-major and V cache
  transposed (channel-major), both f16, so both attention contractions go
  through the standard f16 dot exactly like the reference's classic
  (non-flash) path; rope NORM (llama) and NEOX (qwen2) with
  successive-multiply angle tables.
- **Sampler**: greedy (first argmax, the reference's tie rule),
  temperature/top-k/top-p with a seeded xorshift64* RNG.
- **`siphon run`**: generation with timing, `--dump-logits` (writes the
  same SLGD format as the dev-time reference dumper) and `--golden`
  (teacher-forced replay + logits comparison with the calibrated
  acceptance protocol).

### Verification (real numbers, this sandbox: 2 cores, gcc 14.2, -O2)

- Unit tests: **188,232 checks green under ASan+UBSan** — 86,720 kernel
  checks (per-type dots vs exact double references, f16 roundtrips over
  all 65536 patterns, quantizer invariants, exp-poly vector equivalence)
  and 101,340 threadpool checks (lane invariance, rapid dispatch,
  per-lane scratch isolation, spin-then-park stress), plus the milestone-1
  suites (103 parser + 69 synthetic + 276 fixture cases).
- **`make check-infer`: 6/6 golden comparisons PASS** (TinyLlama-1.1B and
  Qwen2.5-0.5B, Q4_K_M, prompts: france/story/code × 32 generated
  tokens). Goldens are token-serial logits dumps from a dev-time llama.cpp
  reference build pinned to the same AVX2 kernel set. Teacher-forced
  positions 0–1 reproduce the reference **bit-exactly**; beyond that a
  single q8_K rounding boundary diverges and amplifies chaotically —
  the reference's own FA-on vs FA-off paths vary by more (31 logits / 27
  argmax flips on the same run), which calibrates the acceptance band:
  prefill max |logit diff| ≤ 0.6 (short) / ≤ 1.0 (medium), zero real
  argmax flips (reference top-2 gap > 1.0) in the first 24 positions.
  Observed: 0.42/0.51/0.74 (TinyLlama), 0.00/0.61/0.91 (Qwen — the
  france prefill is fully bit-exact), 0 real flips everywhere.
- Decode throughput (2 vCPU sandbox): TinyLlama-1.1B Q4_K_M **~10.3 tok/s**
  (`-t 1`) / ~9.5 tok/s (`-t 2`); Qwen2.5-0.5B Q4_K_M ~9.5 tok/s. The second
  lane buys nothing at this model size on the sandbox's shared vCPUs (the
  per-call dispatch overhead of ~160 small parallel sections per token
  offsets it); the meaningful property is that `-t 1` and `-t 2` produce
  **byte-identical logits**. Batching (milestone 4) is where the threads
  pay off.

### Known limitations (milestone 2)

- Prefill is token-serial (one forward per prompt token); batched GEMM
  prefill arrives with the performance milestone.
- Architectures beyond llama/qwen2 (mistral is nearly free; gemma's
  weighted norms, YaRN scaling and attention softcap are not implemented).
- The kdiff bit-exactness proof covers the AVX2 kernel set (the target
  hardware is AVX2); AVX-512 builds of the reference dispatch to
  different lanes for the f16 dot and q8_0 quantizer and land ~0.4 logits
  apart after 22 layers — inside the acceptance band, but not bit-exact.
- IQ quantization family (IQ2/IQ3/IQ4, milestone 6 scope) is not
  implemented.

## Build

```
make                    # release build -> build/siphon
make debug              # ASan+UBSan build -> build/siphon-asan
make test               # unit tests under sanitizers
make check-infer        # golden logits verification (needs models+goldens,
                        # see "Running the inference verification")
make clean
```

Requires a C11+ compiler, make and pthreads. Nothing else.

## Usage

```
siphon info   [--tensors] <model.gguf>    # inspect a model (instant, mmap)
siphon tokenize [opts] <model.gguf>       # tokenizer verification tool
siphon run [opts] <model.gguf>            # generate / verify against goldens
```

`tokenize` options: `-t <text>` (else stdin), `--ids`, `--no-special`,
`--parse-special`, `--hex` (hex-line in, `"<hex> <ids...>"` out), `--detok`
(ids in, text out).

`run` options: `-p <text>` or `--prompt-file <path>`, `-n <count>`,
`-t <threads>`, `--ctx <n>`, `--temp/--top-k/--top-p/--seed`,
`--dump-logits <file>` (write our logits in the reference's SLGD format),
`--golden <file>` (teacher-forced replay + comparison report).

Later milestones add `serve`, `bench` (currently stubs).

### Running the fixture tests with real models

The fixture tests need the two models above:

```
mkdir -p tests/models
# ~669 MB
curl -L -o tests/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
# ~491 MB
curl -L -o tests/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf'
make test
```

(Or point `SIPHON_TEST_TINYLLAMA` / `SIPHON_TEST_QWEN` at copies elsewhere.)

### Running the inference verification

`make check-infer` replays token-serial logits dumps ("goldens") produced
by a dev-time llama.cpp reference build (AVX2-only kernels, flash
attention off, single thread) and applies the acceptance protocol from
the milestone report. The goldens and models live outside the repo:

```
make check-infer MODELS_DIR=/path/to/models GOLDEN_DIR=/path/to/goldens
# defaults: /tmp/my-project/models and /tmp/my-project/ref/golden
# missing goldens are skipped with a message
```

Reproducing the goldens requires the dev-time reference toolchain (a
llama.cpp checkout built with `-DGGML_CPU_REPACK=OFF -DGGML_NATIVE=OFF
-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON` and the `dump_logits`
companion); the exact recipe is recorded in the project worklog.

## Repository layout

```
include/siphon/   public headers (common, gguf, quant, kernels, threadpool,
                  tensor/backend, model, engine, sampler, unicode, vocab)
src/              implementation (parser, vocab, tokenizers, kernels + AVX2
                  transcription, threadpool, backend, model loader, engine,
                  sampler, CLI)
tests/            unit tests + reference-diffed fixtures
tests/fixtures/   token-id fixtures generated against llama.cpp
scripts/          dev-time generators (unicode tables); not used by the build
```

## Development notes

- `scripts/gen_unicode_tables.py` regenerates `src/unicode_tables.c` from
  Python's unicodedata (dev-time only; the generated C is committed so the
  build never needs Python).
- The tokenizer implementation intentionally mirrors llama.cpp's algorithms,
  including edge-case behavior (whitespace-run splitting, invalid-UTF-8
  replacement, merge-queue tie-breaking), because it is the ecosystem's
  de-facto reference: matching it exactly is what "identical tokenization"
  means in practice for agent harnesses.
