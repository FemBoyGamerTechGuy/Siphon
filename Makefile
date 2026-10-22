# Siphon — raw-C GGUF inference engine
#
# Build requires only a C11+ compiler and make. No external libraries.
#
#   make            release build (build/siphon)
#   make debug      ASan+UBSan build (build/siphon-asan)
#   make test       run unit tests (under sanitizers)
#   make clean

CC      ?= cc
CSTD    ?= c17
OPT     ?= -O2

WARN     = -Wall -Wextra -Werror
BASE     = -std=$(CSTD) $(OPT) -D_POSIX_C_SOURCE=200809L -Iinclude
CFLAGS   ?= $(BASE) $(WARN)
LDFLAGS  ?= -lm -pthread

SRC_DIR := src
OBJ_DIR := build
BIN_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

BIN      := $(BIN_DIR)/siphon
BIN_ASAN := $(BIN_DIR)/siphon-asan

TEST_SRCS := $(wildcard tests/*.c)
TEST_BINS := $(patsubst tests/%.c,$(OBJ_DIR)/test-%,$(TEST_SRCS))
# tests link everything except the CLI entry point
TEST_OBJS := $(filter-out $(OBJ_DIR)/main.o $(OBJ_DIR)/cmd_%.o,$(OBJS))

.PHONY: all debug test check-infer clean

all: $(BIN)

$(BIN): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# sanitizer build: catches memory errors and UB during development
debug: CFLAGS := $(BASE) $(WARN) -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
debug: LDFLAGS := -fsanitize=address,undefined -lm -pthread
debug: $(BIN_ASAN)

$(BIN_ASAN): $(SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

# unit tests, run under sanitizers
test: CFLAGS := $(BASE) $(WARN) -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
test: LDFLAGS := -fsanitize=address,undefined -lm -pthread
test: $(TEST_BINS)
	@fail=0; \
	for t in $(TEST_BINS); do \
	echo "== $$(basename $$t)"; \
	ASAN_OPTIONS=detect_leaks=1 ./$$t || fail=1; \
	done; \
	exit $$fail

$(OBJ_DIR)/test-%: tests/%.c $(TEST_OBJS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -Iinclude -o $@ $< $(TEST_OBJS) $(LDFLAGS)

# inference verification against reference logits dumps (dev-time
# goldens, not shipped in this repo). Set:
#   MODELS_DIR  (default /tmp/my-project/models)
#   GOLDEN_DIR  (default /tmp/my-project/ref/golden)
# Each golden is a token-serial SLGD dump produced by a dev-time
# llama.cpp build (AVX2-only kernels, FA off, single thread). PASS
# requires prefill max |logit diff| within the calibrated band and
# zero real argmax flips in the first 24 positions.
GOLDEN_DIR ?= /tmp/my-project/ref/golden
MODELS_DIR ?= /tmp/my-project/models
GOLDENS := tinyllama-france tinyllama-story tinyllama-code \
           qwen-france qwen-story qwen-code

.PHONY: check-infer
check-infer: $(BIN)
	@fail=0; \
	for g in $(GOLDENS); do \
	if [ "$${g#tinyllama}" != "$$g" ]; then \
	m=$(MODELS_DIR)/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf; \
                else \
	m=$(MODELS_DIR)/qwen2.5-0.5b-instruct-q4_k_m.gguf; \
	fi; \
	echo "== $$g"; \
	if [ ! -f "$(GOLDEN_DIR)/$$g.slgd" ]; then \
	echo "  SKIP (no golden at $(GOLDEN_DIR)/$$g.slgd)"; \
	continue; \
	fi; \
	./$(BIN) run --golden $(GOLDEN_DIR)/$$g.slgd -t 2 $$m || fail=1; \
	done; \
	exit $$fail

clean:
	rm -rf $(OBJ_DIR)
