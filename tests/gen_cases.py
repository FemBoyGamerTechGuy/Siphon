#!/usr/bin/env python3
"""Emit the tokenizer verification battery as hex lines (one per case).

The same battery is fed to `siphon tokenize --hex` and to the dev-time
llama.cpp reference dumper; outputs must match line-for-line.
"""
import sys

CASES = [
    "",                              # empty (only meaningful with special)
    " ",
    "Hello",
    "Hello world",
    "hello WORLD don't stop",
    "I'm we'll you're they've he'd it's she's I'M Don'T iT'S",
    "  double  spaces  ",
    "   triple   spaces   ",
    "\ttabbed\ttext\t",
    "a\nb\r\nc\n\r\n\n d",
    "\n",
    "\r\n\r\n",
    "trailing spaces   ",
    "1234567890 3.14159 1,000,000 42nd 3rd",
    "0x1F 0b101 1e-9",
    "Hello!!! How are you??? ...--— ok.",
    "int main(){return 0;}",
    "def f(x):\n    return x*2\n",
    "    if (x) {\n        return 0;\n    }\n",
    '{"role":"user","content":"hi"}',
    "I said: \"quotes\" and 'single' — see https://example.com/path?q=1&x=2#frag",
    "héllo wörld naïve café résumé Zürich",
    "αβγδε Ωλμ ∑∫π ≈≠≤ √Δ",
    "«»\u201c\u201d\u2018\u2019…–—",
    "你好，世界！这是中文测试。",
    "日本語とEnglishの混在text123",
    "🚀🎉 👨‍👩‍👧‍👦 🇺🇸",
    "русский текст здесь",
    "ᬳᬬ᭄ᬱᬦᬲᬶᬮ᭄",
    "mixedᵃsuperscript ½¾ ⅓ x²",
    "the the the the the the the the",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "Ā Ċ Ġ Ŧ",
    "​",
    "a​b",
    "\x1b[31mred\x1b[0m",
    "a\x00b",
    "\xff\xfe \x80abc\xc3",              # invalid utf-8
    "\xc3\xa9t\xc3\xa9",                 # valid é
    "<|im_start|>user\nhello<|im_end|>",  # special-token text (parse mode)
    "<s></s>[INST] hi [/INST]",
    "I'll be back — isn't it?",
    "e^(i*pi) + 1 = 0",
    "The quick brown fox jumps over the lazy dog. " * 4,
    "\u00a0nbsp\u2009thinsp\u3000ideographic",
    "7777777 00000 1000000",
    "3.5 3,5 ½ ⅓",
]

def main():
    for c in CASES:
        print(c.encode("utf-8", "surrogatepass").hex())

if __name__ == "__main__":
    main()
