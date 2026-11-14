# Colombia – Altered Reviews & Archive Seal

**Flag:** `SIGMOID_DATA_SANITIZATION`

## Data
`reviews.json` holds:
- `trusted_reference`: 1,000 clean reviews
- `samples`: 1,200 suspect reviews (`sample_index` 0–1199)
- `archive_note`: how to open the seals

Each review has `summary`, `review_text`, `rating` (1–5) and a 32-byte hex `archive_tag`. Nothing is empty or duplicated.

The archive note says: sort the altered `sample_index` values, join them with commas, and take the SHA-256. That is a 32-byte mask. XOR every tag with the mask and strip trailing zero bytes. One seal (trusted or sample) holds the message.

Only the right set produces readable text, so the flag itself confirms that the set is correct.

## Step 1 – Text tampering (12 records)
Scanning every non-ASCII character shows two kinds that appear **only in samples, never in the trusted set**:

| Trick | Records |
|---|---|
| Cyrillic look-alike letters (`а с е о`) inside words | 36, 119, 367, 462, 513, 752 |
| Zero-width space `U+200B` inside words | 216, 273, 357, 478, 685, 1199 |

En dashes and curly quotes look suspicious too, but they appear at the same rate in the trusted set. They are dataset formatting, not tampering.

These 12 alone (or either group of 6) open **no** seal, so some records must have been changed without touching the text.

## Step 2 – Label tampering (12 records)
The reviews come from the public **Amazon Musical Instruments 5-core** dataset (McAuley, 2014, 10,261 reviews):
`http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz`

Matching each review to its original by loose text key (undo the look-alike letters, drop punctuation, lowercase):
- all 1,000 trusted and all 1,200 sample texts match an original review
- trusted ratings match the originals 100%
- **12 sample ratings were flipped between 3 and 4**:

| Original 4 → now 3 | Original 3 → now 4 |
|---|---|
| 112, 198, 371, 432, 633, 1061 | 347, 582, 622, 661, 1018, 1112 |

A 3↔4 flip is hard to catch with a classifier, because both sides read as "mild". The text gives some hints though. For example, #371 says *"4 stars is my tops"* but has rating 3, and #432 *"My Long Time Favorite … I love them"* has rating 3.

Apart from the 12 hidden characters, the text diff shows only quote and dash normalisation, which also appears in the trusted set.

## Step 3 – Open the seal
Altered set = 12 text + 12 rating = 24 records:

```
36,112,119,198,216,273,347,357,367,371,432,462,478,513,582,622,633,661,685,752,1018,1061,1112,1199
```

SHA-256 of that string → mask. XOR all 2,200 tags. Exactly one turns into printable ASCII:
**trusted_reference #613 → `SIGMOID_DATA_SANITIZATION`**.

## Reproduce
```bash
curl -L -o data/reviews_Musical_Instruments_5.json.gz \
  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
python3 solve.py data/reviews.json data/reviews_Musical_Instruments_5.json.gz
```

Output:
```
text altered  : [36, 119, 216, 273, 357, 367, 462, 478, 513, 685, 752, 1199]
rating altered: [112, 198, 347, 371, 432, 582, 622, 633, 661, 1018, 1061, 1112]
total altered : 24
seal: trusted #613
FLAG: SIGMOID_DATA_SANITIZATION
```
