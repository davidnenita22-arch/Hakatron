"""France – "Point par point": find the artist's hidden letters in the Paris listings.

Usage:
    python3 solve.py [paris_listings.csv] [inside_airbnb_listings.csv.gz]

Defaults: paris_listings.csv and data/listings.csv.gz
Original public data (Inside Airbnb, Paris):
    https://data.insideairbnb.com/france/ile-de-france/paris/2026-09-16/data/listings.csv.gz

Outputs: labels.csv (listing_id -> letter cluster), letters.png (figure)
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TASK = sys.argv[1] if len(sys.argv) > 1 else "paris_listings.csv"
ORIG = sys.argv[2] if len(sys.argv) > 2 else "data/listings.csv.gz"


def key(lat, lon):
    return list(zip(np.round(lat, 5), np.round(lon, 5)))


def letter_plane(Z):
    """2-D PCA of one letter, with axis signs fixed (largest loading positive)."""
    p = PCA(2, whiten=True).fit(Z)
    comps = p.components_ * np.sign(p.components_[np.arange(2), np.abs(p.components_).argmax(1)])[:, None]
    Y = (Z - p.mean_) @ comps.T / np.sqrt(p.explained_variance_)
    return Y[:, 0], Y[:, 1]


# (flip_x, flip_y, swap_axes) per letter so it reads upright
FLIPS = {"O": (False, False, False), "R": (False, False, False), "A": (False, True, False),
         "N": (False, False, False), "G": (False, False, True), "E": (False, False, True)}


def main():
    df = pd.read_csv(TASK)

    # 1) Separate the artist's dots: real listings exist in the public Inside Airbnb file
    orig = pd.read_csv(ORIG, usecols=["latitude", "longitude"])
    real = set(key(orig.latitude, orig.longitude))
    df["fake"] = [k not in real for k in key(df.latitude, df.longitude)]
    fake = df[df.fake].reset_index(drop=True)
    print(f"listings: {len(df)}  matched real: {(~df.fake).sum()}  artist dots: {len(fake)}")

    # 2) The dots' "colours" = position + what travellers weigh (price, availability, reviews)
    feats = np.c_[fake.latitude, fake.longitude, np.log(fake.price),
                  fake.availability_365, np.log(fake.reviews_per_month)]
    X = StandardScaler().fit_transform(feats)

    # 3) Cluster: each cluster is one letter; grey (-1) = stray real listings
    lab = HDBSCAN(min_cluster_size=20, min_samples=5).fit_predict(X)
    fake["cluster"] = lab
    sizes = pd.Series(lab).value_counts().sort_index()
    print("cluster sizes:", sizes.to_dict())

    # 4) Each letter lies on its own 2-D plane -> PCA to read it.
    #    Cluster numbers can change between runs, so each cluster is named by where
    #    its dots sit in Paris (centre lon, lat). Letters were read from the PCA plots;
    #    PCA axis signs are normalised and each letter gets a fixed orientation
    #    (chosen by eye from all 8 flips) that shows it upright.
    centres = {  # letter: (lon, lat) of the cluster centre, flip_x, flip_y, swap_axes
        "O": ((2.363, 48.866), FLIPS["O"]),
        "R": ((2.374, 48.856), FLIPS["R"]),
        "A": ((2.352, 48.856), FLIPS["A"]),
        "N": ((2.339, 48.874), FLIPS["N"]),
        "G": ((2.301, 48.843), FLIPS["G"]),
        "E": ((2.319, 48.885), FLIPS["E"]),
    }
    word_order = ["O", "R", "A", "N", "G", "E"]

    orient = {}
    for c in sorted(set(lab) - {-1}):
        m = lab == c
        cx, cy = fake.longitude[m].mean(), fake.latitude[m].mean()
        letter = min(centres, key=lambda L: (centres[L][0][0] - cx) ** 2 + (centres[L][0][1] - cy) ** 2)
        orient[c] = (letter, *centres[letter][1])
    assert sorted(v[0] for v in orient.values()) == sorted(word_order), orient

    letter_of = {c: v[0] for c, v in orient.items()}
    fake["letter"] = fake.cluster.map(letter_of).fillna("noise")
    fake[["listing_id", "cluster", "letter"]].to_csv("labels.csv", index=False)

    clusters = sorted(orient, key=lambda c: word_order.index(orient[c][0]))
    fig, axs = plt.subplots(2, 6, figsize=(24, 8.5))
    # top row: where the dots sit in Paris
    for ax, c in zip(axs[0], clusters):
        ax.scatter(df.longitude, df.latitude, s=0.1, c="0.85", lw=0)
        m = fake.cluster == c
        ax.scatter(fake.longitude[m], fake.latitude[m], s=2, c="tab:red", lw=0)
        ax.set_aspect(1 / 0.657)
        ax.set_title(f"cluster {c} on the map", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    # bottom row: the letter on its PCA plane
    for ax, c in zip(axs[1], clusters):
        letter, fx, fy, sw = orient[c]
        a, b = letter_plane(X[lab == c])
        if fx: a = -a
        if fy: b = -b
        if sw: a, b = b, a
        ax.scatter(a, b, s=6, c="tab:blue")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"cluster {c} -> {letter} (n={int((lab == c).sum())})", fontsize=12)
    fig.suptitle("Artist's dots, one letter per cluster: " + "".join(word_order), fontsize=18)
    plt.tight_layout()
    plt.savefig("letters.png", dpi=110)

    word = "".join(word_order)
    print("labels.csv and letters.png written")
    print("WORD:", word)
    print("FLAG:", f"SIGMOID_{word}")


if __name__ == "__main__":
    main()
