/* siphon: command-line entry point */
#include "siphon/common.h"

#include <unistd.h>

int cmd_info(int argc, char **argv);
int cmd_tokenize(int argc, char **argv);

static void usage(FILE *f) {
    fprintf(f,
        "Siphon %s — GGUF inference engine (milestone 1: parser + tokenizers)\n"
        "\n"
        "usage:\n"
        "  siphon info    [--tensors] <model.gguf>   print model metadata\n"
        "  siphon tokenize [options] <model.gguf>   tokenize/detokenize text (debug)\n"
        "\n"
        "planned (later milestones):\n"
        "  siphon run    <model.gguf>   interactive chat        (milestone 2+)\n"
        "  siphon serve  <model.gguf>   OpenAI/Anthropic server (milestone 7)\n"
        "  siphon bench  <model.gguf>   prefill/decode benchmark (milestone 4+)\n"
        "\n"
        "tokenize options:\n"
        "  -t <text>      tokenize this string (default: read stdin)\n"
        "  --ids          print token ids only (space separated)\n"
        "  --no-special   don't add bos/eos special tokens\n"
        "  --parse-special  recognize special tokens in the text\n"
        "  --detok        read ids from stdin, print decoded text\n",
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
    } else if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 ||
               strcmp(cmd, "-h") == 0) {
        usage(stdout);
        rc = 0;
    } else if (strcmp(cmd, "run") == 0 || strcmp(cmd, "serve") == 0 ||
               strcmp(cmd, "bench") == 0) {
        fprintf(stderr, "siphon: '%s' is not implemented yet (see roadmap)\n", cmd);
        rc = 1;
    } else {
        fprintf(stderr, "siphon: unknown command '%s'\n", cmd);
        usage(stderr);
        rc = 1;
    }
    return rc;
}
