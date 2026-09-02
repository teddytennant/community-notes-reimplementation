import numpy as np
import pandas as pd
import pytest

from community_notes.data import prune, to_arrays


def ratings_from(pairs, level="HELPFUL"):
    """Build a tiny ratings frame from (rater, note) pairs."""
    rows = []
    for r, n in pairs:
        rows.append({"noteId": n, "raterParticipantId": r, "helpfulnessLevel": level})
    df = pd.DataFrame(rows)
    df["noteId"] = df["noteId"].astype("int64")
    return df


def test_index_mapping_roundtrip():
    df = ratings_from([("a", 10), ("a", 20), ("b", 10), ("c", 30), ("b", 30)])
    df["helpfulness"] = np.array([1.0, 0.5, 0.0, 1.0, 0.5], dtype="float32")
    out = to_arrays(df, min_note_ratings=1, min_rater_ratings=1)

    assert out["rater_idx"].dtype == np.int32
    assert out["note_idx"].dtype == np.int32
    assert out["rating"].dtype == np.float32
    assert out["note_ids"].dtype == np.int64
    assert len(out["rater_idx"]) == len(out["note_idx"]) == len(out["rating"]) == 5

    seen = set()
    for i in range(5):
        r = out["rater_ids"][out["rater_idx"][i]]
        n = out["note_ids"][out["note_idx"][i]]
        seen.add((r, int(n), float(out["rating"][i])))
    want = set(zip(df.raterParticipantId, df.noteId.astype(int), df.helpfulness.astype(float)))
    assert seen == want
    assert set(out["note_ids"]) == {10, 20, 30}
    assert set(out["rater_ids"]) == {"a", "b", "c"}
    assert out["note_idx"].max() == len(out["note_ids"]) - 1
    assert out["rater_idx"].max() == len(out["rater_ids"]) - 1


def test_helpfulness_mapped_from_level_when_missing():
    df = pd.concat([
        ratings_from([("a", 1)], "HELPFUL"),
        ratings_from([("b", 1)], "SOMEWHAT_HELPFUL"),
        ratings_from([("c", 1)], "NOT_HELPFUL"),
    ], ignore_index=True)
    out = to_arrays(df, min_note_ratings=1, min_rater_ratings=1)
    by_rater = {out["rater_ids"][r]: v for r, v in zip(out["rater_idx"], out["rating"])}
    assert by_rater == {"a": 1.0, "b": 0.5, "c": 0.0}


def test_pruning_iterates_until_stable():
    pairs = [
        ("x", 1), ("w", 1), ("z", 1),
        ("z", 2), ("y", 2),
        ("z", 3),
        ("x", 4), ("w", 4),
    ]
    df = ratings_from(pairs)
    kept = prune(df, min_note_ratings=2, min_rater_ratings=2)
    kept_pairs = set(zip(kept.raterParticipantId, kept.noteId.astype(int)))
    assert kept_pairs == {("x", 1), ("w", 1), ("x", 4), ("w", 4)}
    again = prune(kept, min_note_ratings=2, min_rater_ratings=2)
    assert len(again) == len(kept)


def test_single_pass_would_not_be_enough():
    df = ratings_from([("a", 1), ("b", 1), ("a", 2), ("c", 2), ("c", 3)])
    once = df[df.noteId.map(df.noteId.value_counts()) >= 2]
    once = once[once.raterParticipantId.map(once.raterParticipantId.value_counts()) >= 2]
    assert (once.noteId.value_counts() < 2).any()
    kept = prune(df, min_note_ratings=2, min_rater_ratings=2)
    assert (kept.noteId.value_counts() >= 2).all()
    assert (kept.raterParticipantId.value_counts() >= 2).all()


def test_defaults_prune_sparse_rows():
    raters = [f"r{i}" for i in range(10)]
    pairs = [(r, n) for r in raters for n in range(10)]
    pairs += [("lonely", 0), ("r0", 99)]
    df = ratings_from(pairs)
    out = to_arrays(df)
    assert "lonely" not in set(out["rater_ids"])
    assert 99 not in set(out["note_ids"])
    assert len(out["rating"]) == 100


def test_duplicate_ratings_keep_last():
    df = ratings_from([("a", 1), ("a", 1)])
    df["helpfulness"] = np.array([0.0, 1.0], dtype="float32")
    out = to_arrays(df, min_note_ratings=1, min_rater_ratings=1)
    assert len(out["rating"]) == 1
    assert out["rating"][0] == pytest.approx(1.0)
