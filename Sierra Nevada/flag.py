"""Turn whatever Colibrí leaked into the flag SIGMOID_<PASSWORD>.

Usage:
  python3 flag.py "P-I-C-O _ B-O-L-I-V-A-R"          # spelled out
  python3 flag.py "RAVILOB_OCIP" --reverse            # written backwards
  python3 flag.py "line one / line two" --acrostic    # first letter of each line ('/' or newline)
"""
import re
import sys
import unicodedata


def clean(text):
    # strip accents, keep letters/digits, turn spaces/dashes between words into _
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).upper()
    # spelled form "P-I-C-O _ B-O-..." -> join single letters, '_' or double space = word break
    if re.fullmatch(r"[A-Z0-9](?:[\s\-.,]+[A-Z0-9_]|\s*_\s*)*", text.strip()):
        words = re.split(r"\s*_\s*|\s{2,}", text.strip())
        return "_".join(re.sub(r"[^A-Z0-9]", "", w) for w in words if w)
    return re.sub(r"[^A-Z0-9]+", "_", text).strip("_")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    raw = sys.argv[1]
    if "--acrostic" in sys.argv:
        lines = [l.strip() for l in re.split(r"[/\n]", raw) if l.strip()]
        raw = "".join(l[0] for l in lines)
    if "--reverse" in sys.argv:
        raw = raw[::-1]
    pwd = clean(raw)
    print("password:", pwd)
    print("FLAG:    ", f"SIGMOID_{pwd}")


if __name__ == "__main__":
    main()
