"""Do the seed differences survive running the fit past its stopping rules?

seed_sweep uses mf.fit, which stops on the 1e-7 tolerance. This runs the same
seeds with both stopping rules off, so the loss at the tolerance stop can be
compared with the loss much later.
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from community_notes import mf  # noqa: E402
import fit as fit_script  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 900
SEEDS = [0, 1, 2]

data = fit_script.real_data()
u = np.asarray(data["rater_idx"], dtype=np.int64)
n = np.asarray(data["note_idx"], dtype=np.int64)
r = np.asarray(data["rating"], dtype=np.float64)
counts = mf.ratings_per_note(data)
print(f"{len(r)} ratings, {len(data['rater_ids'])} raters, {len(data['note_ids'])} notes", flush=True)

out = {}
for seed in SEEDS:
    p = mf.init_params(len(data["rater_ids"]), len(data["note_ids"]), 1, seed)
    opt = mf.Adam(mf.ADAM_LR)
    t = time.time()
    prev = mf.loss(p, u, n, r)
    early = None
    for it in range(STEPS):
        g = mf.clip_by_global_norm(mf.gradient(p, u, n, r).arrays(), mf.GRAD_CLIP_NORM)
        p = mf.Params(*opt.step(p.arrays(), g))
        cur = mf.loss(p, u, n, r)
        if early is None and it > 100 and cur > prev:
            early = (it, cur, mf.note_status(mf.flip_factors_for_identification(p), counts).copy())
            print(f"seed {seed}: official rule would stop at step {it}, loss {cur:.6f}", flush=True)
        prev = cur
    p = mf.flip_factors_for_identification(p)
    out[seed] = dict(final_loss=cur, status=mf.note_status(p, counts),
                     intercept=p.note_intercept.copy(),
                     early_status=early[2] if early else None)
    print(f"seed {seed}: {STEPS} steps, loss {cur:.6f}, "
          f"{(out[seed]['status'] == mf.CURRENTLY_RATED_HELPFUL).sum()} helpful, "
          f"{time.time() - t:.0f}s", flush=True)

scored = counts >= mf.MIN_RATINGS


def report(key, label):
    st = np.array([out[s][key] for s in SEEDS])
    unstable = np.array([len(set(st[:, j])) > 1 for j in range(st.shape[1])])
    agree = [float((st[a, scored] == st[b, scored]).mean())
             for a in range(len(SEEDS)) for b in range(a + 1, len(SEEDS))]
    ever = (st == mf.CURRENTLY_RATED_HELPFUL).any(axis=0).sum()
    always = (st == mf.CURRENTLY_RATED_HELPFUL).all(axis=0).sum()
    print(f"{label}: unstable {int((unstable & scored).sum())} of {int(scored.sum())} "
          f"({100 * (unstable & scored).sum() / scored.sum():.2f}%), "
          f"pairwise agreement {min(agree):.4f} to {max(agree):.4f}, "
          f"helpful ever {ever} / always {always}", flush=True)


if out[SEEDS[0]]["early_status"] is not None:
    report("early_status", "at the first loss rise")
report("status", f"after {STEPS} steps")
print("final losses:", {s: round(out[s]["final_loss"], 6) for s in SEEDS}, flush=True)
ints = np.array([out[s]["intercept"] for s in SEEDS])
print(f"intercept std across seeds, median {np.median(ints[:, scored].std(axis=0)):.5f} "
      f"p95 {np.percentile(ints[:, scored].std(axis=0), 95):.5f}", flush=True)
