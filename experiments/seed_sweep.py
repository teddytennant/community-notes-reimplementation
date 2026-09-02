"""Fit the same ratings under several seeds and see which note statuses move.

Initialization is random, so a note near a threshold can land on either side
of it. This reports which notes flip, how that depends on how many ratings a
note has, and how far the intercepts spread. Output goes to experiments/out/.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from community_notes import mf  # noqa: E402
import fit as fit_script  # noqa: E402

OUT = ROOT / "experiments" / "out"
BINS = [5, 10, 20, 50, 100, 200, 1_000_000]


def run(data, seeds, optimizer):
    n_notes = len(data["note_ids"])
    intercepts = np.zeros((len(seeds), n_notes))
    factors = np.zeros((len(seeds), n_notes))
    statuses = np.empty((len(seeds), n_notes), dtype=object)
    counts = mf.ratings_per_note(data)
    losses = []
    for i, seed in enumerate(seeds):
        t = time.time()
        params, history = mf.fit(data, seed=seed, optimizer=optimizer)
        intercepts[i] = params.note_intercept
        factors[i] = params.note_factor[:, 0]
        statuses[i] = mf.note_status(params, counts)
        losses.append(float(history[-1]))
        print(f"seed {seed}: {len(history) - 1} steps, loss {history[-1]:.5f}, "
              f"{(statuses[i] == mf.CURRENTLY_RATED_HELPFUL).sum()} helpful, "
              f"{(statuses[i] == mf.CURRENTLY_RATED_NOT_HELPFUL).sum()} not helpful, "
              f"{time.time() - t:.1f}s")
    return intercepts, factors, statuses, counts, losses


def analyze(intercepts, factors, statuses, counts, losses, seeds):
    n_seeds, n_notes = statuses.shape
    scored = counts >= mf.MIN_RATINGS
    stable = np.array([len(set(statuses[:, j])) == 1 for j in range(n_notes)])
    ever_helpful = (statuses == mf.CURRENTLY_RATED_HELPFUL).any(axis=0)
    always_helpful = (statuses == mf.CURRENTLY_RATED_HELPFUL).all(axis=0)

    summary = {
        "seeds": list(map(int, seeds)),
        "notes": int(n_notes),
        "notes_with_enough_ratings": int(scored.sum()),
        "final_losses": losses,
        "loss_spread": float(np.max(losses) - np.min(losses)),
        "unstable_notes": int((~stable & scored).sum()),
        "unstable_fraction": float((~stable & scored).sum() / max(scored.sum(), 1)),
        "helpful_in_any_seed": int(ever_helpful.sum()),
        "helpful_in_every_seed": int(always_helpful.sum()),
        "intercept_std_median": float(np.median(intercepts[:, scored].std(axis=0))),
        "intercept_std_p95": float(np.percentile(intercepts[:, scored].std(axis=0), 95)),
        "by_rating_count": [],
    }
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = scored & (counts >= lo) & (counts < hi)
        if m.sum() == 0:
            continue
        summary["by_rating_count"].append({
            "ratings_from": lo,
            "ratings_below": hi,
            "notes": int(m.sum()),
            "unstable_fraction": float((~stable & m).sum() / m.sum()),
            "intercept_std_median": float(np.median(intercepts[:, m].std(axis=0))),
        })
    pair = []
    for a in range(n_seeds):
        for b in range(a + 1, n_seeds):
            pair.append(float((statuses[a, scored] == statuses[b, scored]).mean()))
    summary["pairwise_status_agreement_mean"] = float(np.mean(pair)) if pair else 1.0
    summary["pairwise_status_agreement_min"] = float(np.min(pair)) if pair else 1.0
    return summary, stable


def plots(intercepts, factors, counts, stable, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = counts >= mf.MIN_RATINGS
    mean_b = intercepts.mean(axis=0)
    std_b = intercepts.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(counts[scored], std_b[scored], s=3, alpha=0.3, color="#4a6fa5")
    ax.set_xscale("log")
    ax.set_xlabel("ratings on the note")
    ax.set_ylabel("std of note intercept across seeds")
    ax.set_title("Seed sensitivity shrinks with more ratings")
    fig.tight_layout()
    fig.savefig(OUT / "intercept_std_vs_ratings.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mean_b[scored], bins=120, color="#4a6fa5")
    ax.axvline(mf.CRH_THRESHOLD, color="#c44", ls="--", label="helpful threshold 0.40")
    ax.axvline(mf.CRNH_INTERCEPT, color="#888", ls="--", label="not helpful base -0.05")
    ax.set_xlabel("mean note intercept across seeds")
    ax.set_ylabel("notes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "intercept_hist.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    unstable = scored & ~stable
    ax.scatter(mean_b[scored & stable], factors.mean(axis=0)[scored & stable], s=2, alpha=0.2, color="#999", label="stable")
    ax.scatter(mean_b[unstable], factors.mean(axis=0)[unstable], s=6, color="#c44", label="status depends on seed")
    ax.axvline(mf.CRH_THRESHOLD, color="#c44", ls="--", lw=0.8)
    ax.set_xlabel("note intercept (helpfulness)")
    ax.set_ylabel("note factor (which side likes it)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "intercept_vs_factor.png", dpi=150)

    rows = summary["by_rating_count"]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"{r['ratings_from']}+" if r["ratings_below"] >= 1_000_000 else f"{r['ratings_from']}-{r['ratings_below'] - 1}" for r in rows]
        ax.bar(labels, [100 * r["unstable_fraction"] for r in rows], color="#4a6fa5")
        ax.set_xlabel("ratings on the note")
        ax.set_ylabel("% of notes whose status changes with the seed")
        fig.tight_layout()
        fig.savefig(OUT / "unstable_by_ratings.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--optimizer", choices=["adam", "gd"], default="adam")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = fit_script.synthetic_data() if args.synthetic else fit_script.real_data()
    print(f"{len(data['rating'])} ratings, {len(data['rater_ids'])} raters, {len(data['note_ids'])} notes")
    seeds = list(range(args.seeds))
    intercepts, factors, statuses, counts, losses = run(data, seeds, args.optimizer)
    summary, stable = analyze(intercepts, factors, statuses, counts, losses, seeds)

    tag = "synthetic" if args.synthetic else "real"
    np.savez_compressed(OUT / f"seed_sweep_{tag}.npz", note_ids=data["note_ids"],
                        intercepts=intercepts, factors=factors, statuses=statuses.astype(str), counts=counts)
    (OUT / f"seed_sweep_{tag}.json").write_text(json.dumps(summary, indent=2))
    plots(intercepts, factors, counts, stable, summary)

    print(json.dumps({k: v for k, v in summary.items() if k != "by_rating_count"}, indent=2))
    for r in summary["by_rating_count"]:
        print(f"ratings {r['ratings_from']:>4}-{r['ratings_below']:<7} notes {r['notes']:>6}  "
              f"unstable {100 * r['unstable_fraction']:5.1f}%  intercept std {r['intercept_std_median']:.4f}")


if __name__ == "__main__":
    main()
