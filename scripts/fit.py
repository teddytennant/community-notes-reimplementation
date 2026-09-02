"""Fit the note scoring model and report how many notes land in each status.

With data/ratings.parquet present this loads the real ratings through
community_notes.data; otherwise pass --synthetic to run on a planted population.
If data/status.parquet exists the fitted note intercepts are compared with
the official core model's.
"""
import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from community_notes import mf  # noqa: E402

RATINGS_PATH = ROOT / "data" / "ratings.parquet"
STATUS_PATH = ROOT / "data" / "status.parquet"


def synthetic_data(seed=0, n_raters=400, n_notes=200, density=0.15):
    """Ratings from a planted one-factor model, snapped to {0, 0.5, 1}."""
    rng = np.random.default_rng(seed)
    truth = mf.Params(
        mu=np.array(0.3),
        rater_intercept=rng.normal(0, 0.1, n_raters),
        note_intercept=rng.normal(0, 0.4, n_notes),
        rater_factor=rng.normal(0, 1.0, (n_raters, 1)),
        note_factor=rng.normal(0, 0.5, (n_notes, 1)),
    )
    u, n = np.nonzero(rng.random((n_raters, n_notes)) < density)
    r = mf.predict(truth, u, n) + rng.normal(0, 0.1, len(u))
    r = np.round(np.clip(r, 0, 1) * 2) / 2
    return dict(
        rater_idx=u.astype(np.int32),
        note_idx=n.astype(np.int32),
        rating=r.astype(np.float32),
        rater_ids=np.arange(n_raters),
        note_ids=np.arange(n_notes),
    )


def real_data(min_note_ratings=5, min_rater_ratings=10):
    from community_notes import data as nd

    return nd.to_arrays(nd.load_ratings(), min_note_ratings, min_rater_ratings)


def compare_with_official(note_ids, params, status):
    """Report agreement with the official core model.

    The dump carries official statuses but not fitted intercepts, so the
    comparison is on status. currentCoreStatus is the core model's own
    verdict, currentStatus is what users see after the other models weigh in.
    """
    import pandas as pd

    official = pd.read_parquet(STATUS_PATH)
    official["noteId"] = official["noteId"].astype(np.int64)
    col = next((c for c in ("currentCoreStatus", "coreRatingStatus", "currentStatus") if c in official.columns), None)
    if col is None or "noteId" not in official.columns:
        print("status parquet has no status column I recognise; its columns are:")
        for c in official.columns:
            print("  ", c)
        return
    ours = pd.DataFrame(
        {
            "noteId": np.asarray(note_ids).astype(np.int64),
            "intercept": params.note_intercept,
            "factor": params.note_factor[:, 0],
            "status": status,
        }
    )
    both = ours.merge(official[["noteId", col]], on="noteId", how="inner").dropna(subset=[col])
    print(f"notes in both: {len(both)} of {len(ours)} fitted, official column {col}")
    for c in ("coreNoteIntercept", "coreNoteFactor1"):
        if c in official.columns:
            m = ours.merge(official[["noteId", c]], on="noteId").dropna(subset=[c])
            key = "intercept" if "Intercept" in c else "factor"
            print(f"pearson r, note {key} vs {c}: {np.corrcoef(m[key], m[c])[0, 1]:.4f}")
    print("raw official statuses vs ours:")
    print(pd.crosstab(both[col], both["status"]).to_string())
    collapsed = both[col].where(  # FIRM_REJECT and NEEDS_YOUR_HELP are not core threshold outputs
        both[col].isin([mf.CURRENTLY_RATED_HELPFUL, mf.CURRENTLY_RATED_NOT_HELPFUL]),
        mf.NEEDS_MORE_RATINGS,
    )
    print(f"three way agreement on {len(both)} notes: {(both['status'] == collapsed).mean():.4f}")
    for label in (mf.CURRENTLY_RATED_HELPFUL, mf.CURRENTLY_RATED_NOT_HELPFUL):
        ours_l = both["status"] == label
        theirs_l = collapsed == label
        tp = int((ours_l & theirs_l).sum())
        prec = tp / max(int(ours_l.sum()), 1)
        rec = tp / max(int(theirs_l.sum()), 1)
        print(f"{label}: we say {int(ours_l.sum())}, official says {int(theirs_l.sum())}, "
              f"precision {prec:.3f}, recall {rec:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", action="store_true", help="use a planted population")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=1, help="latent dimensions")
    ap.add_argument("--optimizer", choices=["adam", "gd"], default="adam")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=mf.CONVERGENCE)
    ap.add_argument("--min-note-ratings", type=int, default=5, help="drop notes with fewer ratings")
    ap.add_argument("--min-rater-ratings", type=int, default=10,
                    help="drop raters with fewer ratings in the slice; the official scorer counts all time")
    args = ap.parse_args()

    if args.synthetic:
        data = synthetic_data(args.seed)
    elif RATINGS_PATH.exists():
        data = real_data(args.min_note_ratings, args.min_rater_ratings)
    else:
        sys.exit(f"{RATINGS_PATH} not found; pass --synthetic")
    print(f"{len(data['rating'])} ratings, {len(data['rater_ids'])} raters, {len(data['note_ids'])} notes")

    t = time.time()
    params, history = mf.fit(
        data, k=args.k, seed=args.seed, optimizer=args.optimizer, lr=args.lr,
        max_iters=args.max_iters, tol=args.tol,
    )
    print(f"{len(history) - 1} steps in {time.time() - t:.1f}s, loss {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"global intercept {float(params.mu):.4f}")

    status = mf.note_status(params, mf.ratings_per_note(data))
    for name, count in zip(*np.unique(status, return_counts=True)):
        print(f"{name:28s} {count}")

    if not args.synthetic and STATUS_PATH.exists():
        compare_with_official(data["note_ids"], params, status)


if __name__ == "__main__":
    main()
