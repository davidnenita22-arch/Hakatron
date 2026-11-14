# France – Point par point

**Flag:** `SIGMOID_ORANGE`

Files: [`solve.py`](solve.py) (code) · [`letters.png`](letters.png) (figure) · [`labels.csv`](labels.csv) (listing_id → cluster → letter)

![letters](letters.png)

## 1. Separate the artist's dots from the city
`paris_listings.csv` holds 28,108 listings with 18 columns. The listings are internally consistent (review counts, room types, neighbourhoods), so the fake rows can't be spotted from the data alone.

The real listings come from the public **Inside Airbnb Paris** file:
`https://data.insideairbnb.com/france/ile-de-france/paris/2026-09-16/data/listings.csv.gz`

Matching on exact (latitude, longitude):
- **26,163** rows match a real Inside Airbnb listing
- **1,945** rows don't. These are the artist's dots.

On the map, the 1,945 dots form strokes across central Paris instead of being scattered.

## 2. Find the letters ("their colours came from what every traveller weighs")
Only 6 columns carry structure among the fake dots: a dot's neighbours in feature space predict them almost perfectly. These are `latitude`, `longitude`, `price`, `availability_365`, `reviews_per_month` and `number_of_reviews_ltm`. The other columns are copied noise.

Features used: `lat, lon, log(price), availability_365, log(reviews_per_month)`, standardised.

**HDBSCAN** (`min_cluster_size=20, min_samples=5`) gives **6 clusters** of about 300 dots each, plus 1 noise point (grey).

## 3. Look at each letter
Each cluster lies on its own **2-D plane** inside the 5-D feature space. The 3rd PCA component explains only about 2% of the variance. So a 2-D PCA of each cluster draws its letter. PCA signs and axes are arbitrary, so each letter is shown in the flip/rotation that reads upright. The letters are also slightly sheared, because each one was embedded with its own linear map.

| cluster centre (lon, lat) | letter |
|---|---|
| 2.363, 48.866 | **O** (clean ring) |
| 2.374, 48.856 | **R** |
| 2.352, 48.856 | **A** |
| 2.339, 48.874 | **N** |
| 2.301, 48.843 | **G** |
| 2.319, 48.885 | **E** |

Letters: O, R, A, N, G, E → **ORANGE**, a colour, which fits the pointillism story.

## Reproduce
```bash
curl -L -o data/listings.csv.gz https://data.insideairbnb.com/france/ile-de-france/paris/2026-09-16/data/listings.csv.gz
python3 solve.py paris_listings.csv data/listings.csv.gz
```
```
listings: 28108  matched real: 26163  artist dots: 1945
cluster sizes: {-1: 1, 0: 297, 1: 337, 2: 314, 3: 304, 4: 337, 5: 355}
labels.csv and letters.png written
WORD: ORANGE
FLAG: SIGMOID_ORANGE
```

## Caveats
- No column in the data fixes the letter order. ORANGE is the only common word made from these six letters; the anagrams ONAGER and ORGANE are possible but much less likely.
- R and A are the most sheared letters. O, N, G and E are unambiguous.
- Inside Airbnb updates its snapshot. Matching worked with the 2026-09-16 file; an older or newer file may match slightly fewer real rows.
