"""Colombia task - find altered review records and open the archive seal.

Usage:
    python solve.py [reviews.json] [original.json.gz]

Defaults: data/reviews.json and data/reviews_Musical_Instruments_5.json.gz
Original public dataset (Amazon Musical Instruments 5-core, McAuley 2014):
    http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
"""
import gzip
import hashlib
import json
import re
import sys

REVIEWS = sys.argv[1] if len(sys.argv) > 1 else "data/reviews.json"
ORIGINAL = sys.argv[2] if len(sys.argv) > 2 else "data/reviews_Musical_Instruments_5.json.gz"

# Cyrillic look-alikes found in the data -> Latin
HOMOGLYPHS = {"а": "a", "с": "c", "е": "e", "о": "o"}
HIDDEN = set(HOMOGLYPHS) | {"​"}  # + zero-width space


def key(text):
    """Loose text key: undo homoglyphs, drop punctuation/space, lowercase."""
    text = "".join(HOMOGLYPHS.get(c, c) for c in text)
    return re.sub(r"\W+", "", text.lower())


def main():
    data = json.load(open(REVIEWS, encoding="utf-8"))
    samples = data["samples"]

    # 1) Text tampering: invisible / look-alike characters (never present in trusted set)
    text_altered = {s["sample_index"] for s in samples
                    if HIDDEN & set(s["summary"] + s["review_text"])}

    # 2) Label tampering: rating differs from the original public review
    original = {}
    with gzip.open(ORIGINAL, "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            original.setdefault(key(o["reviewText"]), set()).add(int(o["overall"]))
    rating_altered = {s["sample_index"] for s in samples
                      if key(s["review_text"]) in original
                      and s["rating"] not in original[key(s["review_text"])]}

    altered = sorted(text_altered | rating_altered)
    print("text altered  :", sorted(text_altered))
    print("rating altered:", sorted(rating_altered))
    print("total altered :", len(altered))

    # 3) Archive note: SHA-256 of "i1,i2,..." is a 32-byte XOR mask
    fingerprint = ",".join(map(str, altered))
    mask = hashlib.sha256(fingerprint.encode("utf-8")).digest()
    seals = [("trusted", i, r["archive_tag"]) for i, r in enumerate(data["trusted_reference"])]
    seals += [("sample", s["sample_index"], s["archive_tag"]) for s in samples]
    for src, i, tag in seals:
        msg = bytes(a ^ b for a, b in zip(bytes.fromhex(tag), mask)).rstrip(b"\0")
        if msg and all(32 <= c < 127 for c in msg):
            print(f"seal: {src} #{i}")
            print("FLAG:", msg.decode())
            return
    print("no seal opened")


if __name__ == "__main__":
    main()
