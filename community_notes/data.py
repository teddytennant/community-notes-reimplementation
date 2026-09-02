"""Load the data slice under data/ and turn ratings into index arrays.

The parquet files are produced by scripts/build_slice.py. See data/README.md
for what each one holds.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def _path(name, path):
    return path if path is not None else os.path.join(DATA, name)


def load_ratings(path=None):
    """Ratings for the slice: one row per (rater, note) rating."""
    return pd.read_parquet(_path("ratings.parquet", path))


def load_notes(path=None):
    """Notes in the slice with author, tweet, classification and text."""
    return pd.read_parquet(_path("notes.parquet", path))


def load_status(path=None):
    """noteStatusHistory rows for the slice notes, the official outcomes."""
    return pd.read_parquet(_path("status.parquet", path))


def prune(ratings, min_note_ratings=5, min_rater_ratings=10):
    """Drop sparse notes and raters until both thresholds hold at once.

    Removing a rater can push a note under its minimum and the other way
    round, so the two filters are applied in turn until nothing changes. This
    mirrors how the official scorer trims the matrix before factorization.
    """
    df = ratings
    while True:
        n0 = len(df)
        note_counts = df["noteId"].map(df["noteId"].value_counts())
        df = df[note_counts >= min_note_ratings]
        rater_counts = df["raterParticipantId"].map(df["raterParticipantId"].value_counts())
        df = df[rater_counts >= min_rater_ratings]
        if len(df) == n0:
            return df


def to_arrays(ratings, min_note_ratings=5, min_rater_ratings=10):
    """Convert a ratings DataFrame to dense index arrays for factorization.

    Returns a dict with rater_idx and note_idx (int32, same length), rating
    (float32, the helpfulness column, or helpfulnessLevel mapped if the float
    column is missing), note_ids (int64, position gives note_idx) and
    rater_ids (object, position gives rater_idx). Rows with no usable rating
    are dropped before pruning.
    """
    df = ratings
    if "helpfulness" not in df.columns:
        m = {"NOT_HELPFUL": 0.0, "SOMEWHAT_HELPFUL": 0.5, "HELPFUL": 1.0}
        df = df.assign(helpfulness=df["helpfulnessLevel"].map(m).astype("float32"))
    df = df[df["helpfulness"].notna()]
    df = df.drop_duplicates(["raterParticipantId", "noteId"], keep="last")
    df = prune(df, min_note_ratings, min_rater_ratings)

    note_ids, note_idx = np.unique(df["noteId"].to_numpy(dtype=np.int64),
                                   return_inverse=True)
    rater_ids, rater_idx = np.unique(df["raterParticipantId"].to_numpy(dtype=object),
                                     return_inverse=True)
    return {
        "rater_idx": rater_idx.astype(np.int32),
        "note_idx": note_idx.astype(np.int32),
        "rating": df["helpfulness"].to_numpy(dtype=np.float32),
        "note_ids": note_ids,
        "rater_ids": rater_ids,
    }
