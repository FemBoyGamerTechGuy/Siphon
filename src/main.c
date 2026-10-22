/* siphon: command-line entry point */
#include "siphon/common.h"

#include <unistd.h>

int cmd_info(int argc, char **argv);
int cmd_tokenize(int argc, char **argv);
int cmd_run(int argc, char **argv);

static void usage(FILE *f) {
    fprintf(f,
        "Siphon %s — GGUF inference engine (milestone 2: CPU inference, llama + qwen2)\n"
        "\n"
        "usage:\n"
        "  siphon info    [--tensors] <model.gguf>   print model metadata\n"
        "  siphon tokenize [options] <model.gguf>   tokenize/detokenize text (debug)\n"
        "  siphon run [options] <model.gguf>        generate / verify against goldens\n"
        "\n"
        "planned (later milestones):\n"
        "  siphon serve  <model.gguf>   OpenAI/Anthropic server (milestone 7)\n"
        "  siphon bench  <model.gguf>   prefill/decode benchmark (milestone 4+)\n"
        "\n"
        "tokenize options:\n"
        "  -t <text>      tokenize this string (default: read stdin)\n"
        "  --ids          print token ids only (space separated)\n"
        "  --no-special   don't add bos/eos special tokens\n"
        "  --parse-special  recognize special tokens in the text\n"
        "  --detok        read ids from stdin, print decoded text\n"
        "\n"
        "run options:\n"
        "  -p <text>          prompt string (or --prompt-file <path>)\n"
        "  -n <count>         tokens to generate (default 64)\n"
        "  -t <threads>       worker threads (default 2)\n"
        "  --ctx <n>          context slots (default 512)\n"
        "  --temp <f>         sampling temperature (0 = greedy, default)\n"
        "  --top-k <n> / --top-p <f> / --seed <n>   sampling controls\n"
        "  --golden <file>    teacher-forced replay + logits comparison\n"
        "  --dump-logits <f>  write our logits in SLGD format for diffing\n",
        SIPHON_VERSION);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(stderr);
        return 1;
    }
    const char *cmd = argv[1];
    int rc;
    if (strcmp(cmd, "info") == 0) {
        rc = cmd_info(argc - 2, argv + 2);
    } else if (strcmp(cmd, "tokenize") == 0) {
        rc = cmd_tokenize(argc - 2, argv + 2);
    } else if (strcmp(cmd, "run") == 0) {
        rc = cmd_run(argc - 2, argv + 2);
    } else if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 ||
               strcmp(cmd, "-h") == 0) {
        usage(stdout);
        rc = 0;
    } else if (strcmp(cmd, "serve") == 0 || strcmp(cmd, "bench") == 0) {
        fprintf(stderr, "siphon: '%s' is not implemented yet (see roadmap)\n", cmd);
        rc = 1;
    } else {
        fprintf(stderr, "siphon: unknown command '%s'\n", cmd);
        usage(stderr);
        rc = 1;
    }
    return rc;
}
