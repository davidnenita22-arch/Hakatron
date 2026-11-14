#!/usr/bin/env python3
"""Source-assisted recovery of the Point par point accommodation-data artwork.

The letter names are human readings of the generated scatterplots. Clustering,
row selection, and reading order are calculated from the data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

os.environ.setdefault("MPLCONFIGDIR", "/tmp/point-par-point-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REFERENCE_URL = (
    "https://data.insideairbnb.com/france/ile-de-france/paris/"
    "2026-09-16/data/listings.csv.gz"
)
FEATURES = ["longitude", "latitude", "price", "availability_365"]
READING = "NEURON"  # Human transcription, checked against letters.png.
# PCA axes have arbitrary orientation/sign. These viewing transformations affect
# only the displayed coordinates, never cluster membership or the reading order.
ANGLES = [330, 320, 270, 90, 0, 30]
FLIP_VERTICAL = [False, False, False, True, False, False]
COLORS = ["#166baf", "#00968b", "#759b22", "#d29216", "#c95052", "#8757b2"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_matches(data: pd.DataFrame, reference_path: Path) -> np.ndarray:
    """Match complete records, allowing only the challenge's rounding/renaming.

    listing_id is deliberately omitted: challenge identifiers were replaced.
    Dropping duplicate reference signatures makes this a many-to-one join.
    """
    keys = [c for c in data.columns if c != "listing_id"]
    source_names = [
        "neighbourhood_cleansed" if c == "neighbourhood" else c for c in keys
    ]
    reference = pd.read_csv(reference_path, usecols=source_names, low_memory=False)
    reference = reference.rename(columns={"neighbourhood_cleansed": "neighbourhood"})
    reference["price"] = pd.to_numeric(
        reference.price.str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )
    reference["host_is_superhost"] = reference.host_is_superhost.map({"t": True, "f": False})
    for column in ["latitude", "longitude"]:
        reference[column] = reference[column].round(5)
    for column in ["price", "review_scores_rating", "reviews_per_month"]:
        reference[column] = reference[column].round(2)
    lookup = reference[keys].drop_duplicates()
    joined = (
        data.reset_index(names="row_index")
        .merge(lookup, on=keys, how="left", indicator=True, validate="many_to_one")
        .sort_values("row_index")
    )
    assert len(joined) == len(data)
    return joined["_merge"].eq("both").to_numpy()


def solve(input_path: Path, reference_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    assert data.listing_id.is_unique, "Expected one row per listing identifier."
    matched = source_matches(data, reference_path)
    candidate_indices = np.flatnonzero(~matched)
    candidates = data.iloc[candidate_indices]
    scaled = StandardScaler().fit_transform(candidates[FEATURES])
    raw_labels = HDBSCAN(
        min_cluster_size=40, min_samples=6, cluster_selection_method="eom", copy=True
    ).fit_predict(scaled)
    groups = sorted(
        set(raw_labels) - {-1},
        key=lambda label: candidates.loc[raw_labels == label, "price"].mean(),
    )
    if len(groups) != len(READING):
        raise RuntimeError(f"Expected six readable groups; recovered {len(groups)}.")

    labels = data[["listing_id", *FEATURES]].copy()
    labels.insert(0, "row_index", np.arange(len(data)))
    labels["source_match"] = matched
    labels["cluster"] = -1
    labels["letter"] = ""
    summary = []
    projections = []
    for order, (raw_label, letter) in enumerate(zip(groups, READING), start=1):
        selected = raw_labels == raw_label
        indices = candidate_indices[selected]
        rows = data.iloc[indices]
        local_scaled = StandardScaler().fit_transform(rows[FEATURES])
        pca = PCA(n_components=2, whiten=True, svd_solver="full")
        points = pca.fit_transform(local_scaled)
        theta = np.deg2rad(ANGLES[order - 1])
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        displayed = points @ rotation.T
        if FLIP_VERTICAL[order - 1]:
            displayed[:, 1] *= -1
        labels.loc[indices, "cluster"] = order
        labels.loc[indices, "letter"] = letter
        projections.append((points, displayed, indices))
        summary.append({
            "cluster": order,
            "letter": letter,
            "count": len(indices),
            "mean_price": float(rows.price.mean()),
            "mean_availability_365": float(rows.availability_365.mean()),
            "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
            "view_rotation_degrees": ANGLES[order - 1],
            "view_flip_vertical": FLIP_VERTICAL[order - 1],
        })

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    for oriented, filename in [(True, "letters.png"), (False, "letters_unoriented.png")]:
        fig, axes = plt.subplots(1, 6, figsize=(17, 4.3), layout="constrained")
        for i, (ax, (points, displayed, indices), record) in enumerate(zip(axes, projections, summary)):
            p = displayed if oriented else points
            ax.scatter(p[:, 0], p[:, 1], s=10, color=COLORS[i], alpha=.85, linewidths=0)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{i + 1} · {record['letter']}\n{record['count']} points", pad=15)
        fig.suptitle(
            "Point par point — letters ordered by increasing mean price"
            if oriented else "The same six clusters — PCA coordinates before viewing rotations",
            fontsize=17,
        )
        fig.savefig(output / filename, dpi=180, facecolor="white")
        if oriented:
            fig.savefig(output / "letters.svg", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for ax, x, y in [(axes[0], "longitude", "latitude"), (axes[1], "price", "availability_365")]:
        background = labels.cluster.eq(-1)
        ax.scatter(data.loc[background, x], data.loc[background, y], s=2, color="#d6dadd", alpha=.55, rasterized=True)
        for i, (_, _, indices) in enumerate(projections):
            ax.scatter(data.loc[indices, x], data.loc[indices, y], s=7, color=COLORS[i], label=f"{i+1}: {READING[i]}", linewidths=0)
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_aspect(1 / np.cos(np.deg2rad(data.latitude.mean())))
    axes[0].set_title("Recovered artwork in Paris")
    axes[1].set_title("The same points in booking attributes")
    axes[1].legend(frameon=False, loc="upper right", markerscale=2)
    fig.savefig(output / "recovery_overview.png", dpi=180, facecolor="white")
    plt.close(fig)

    labels.to_csv(output / "labels.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "cluster_summary.csv", index=False)
    result = {
        "flag": f"SIGMOID_{READING}",
        "method": "Full-record public-source matching, HDBSCAN in four dimensions, local PCA, human glyph reading",
        "input_sha256": sha256(input_path),
        "reference_url": REFERENCE_URL,
        "reference_sha256": sha256(reference_path),
        "rows": len(data),
        "exact_source_matches": int(matched.sum()),
        "candidate_rows": len(candidate_indices),
        "candidate_noise_rows": int((raw_labels == -1).sum()),
        "artwork_rows": int(labels.cluster.gt(0).sum()),
        "background_rows": int(labels.cluster.eq(-1).sum()),
        "clusters": summary,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    # Verify the exported labels preserve the input order and reconstruct counts.
    saved = pd.read_csv(output / "labels.csv", keep_default_na=False)
    assert np.array_equal(saved.listing_id.to_numpy(), data.listing_id.to_numpy())
    assert saved.row_index.tolist() == list(range(len(data)))
    assert saved.groupby("cluster").size().to_dict() == labels.groupby("cluster").size().to_dict()
    assert not labels.loc[matched, "cluster"].gt(0).any()
    assert labels.cluster.gt(0).sum() + labels.cluster.eq(-1).sum() == len(data)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/paris_listings.csv"))
    parser.add_argument("--reference", type=Path, default=Path("data/insideairbnb_2026-09-16.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--download-reference", action="store_true")
    args = parser.parse_args()
    if not args.reference.exists():
        if not args.download_reference:
            parser.error("Reference file missing; pass --download-reference or provide --reference.")
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(REFERENCE_URL, args.reference)
    print(json.dumps(solve(args.input, args.reference, args.output), indent=2))
