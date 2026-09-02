# Results

All numbers from the June 1 to 14, 2026 slice described in `data/README.md`.
Pruning raters with fewer than 10 ratings and notes with fewer than 5, the way
the official scorer does, leaves 1,227,354 ratings from 46,183 raters on 24,371
notes. One fit is 143 Adam steps and about 30 seconds.

## Does the reimplementation match the real thing

Compared against `currentCoreStatus` on the 19,336 notes that appear in both:

| | Helpful | Not Helpful |
| --- | --- | --- |
| ours | 647 | 202 |
| official | 1,746 | 760 |
| precision | 0.73 | 0.71 |
| recall | 0.27 | 0.19 |

Three way agreement is 0.89. When this model calls a note Helpful it is
usually right, but it finds well under half of the notes the official scorer
found. That gap is the parts left out, listed in `docs/algorithm.md`. The big
ones are that the official core scorer fits five times with rater filtering
between rounds, and that a rater's rating count is computed over all time
rather than inside a two week window, so raters who look sparse here are not
sparse to the real scorer.

## How much does the seed matter

`seed_sweep.py` fits the same ratings under seeds 0 through 9. Every fit
converges to about the same loss, spread 0.0032, so these are all good fits by
the objective. The statuses still move.

- 708 of 24,371 notes, 2.9%, get a different status depending on the seed.
- 1,066 notes are Helpful under at least one seed. Only 598 are Helpful under
  all ten. Of the notes that could earn the badge, 44% do not earn it every time.
- Mean status agreement between any two seeds is 0.988, worst pair 0.980.
- Median intercept spread across seeds is 0.004, but the 95th percentile is
  0.036, which is a tenth of the distance from zero to the 0.40 threshold.

Instability is not spread evenly. It rises with rating count, from 1.2% for
notes with 5 to 10 ratings to 6.3% for notes with 50 to 100:

| ratings on the note | notes | status depends on seed |
| --- | --- | --- |
| 5 to 9 | 6,544 | 1.2% |
| 10 to 19 | 6,104 | 1.5% |
| 20 to 49 | 6,006 | 3.5% |
| 50 to 99 | 2,858 | 6.3% |
| 100 to 199 | 1,688 | 5.9% |
| 200+ | 1,171 | 4.1% |

That is backwards from what you would expect. More ratings should pin a note
down. The reason is that the notes with few ratings are nowhere near either
threshold, so nothing can push them over. Notes with many ratings are the ones
actually being contested, and a contested note sits near 0.40 where a small
change in the fit decides the outcome. `out/intercept_vs_factor.png` shows the
unstable notes clustered on the threshold line.

Reproduce with:

    .venv/bin/python experiments/seed_sweep.py --seeds 10

Plots and the raw per-seed intercepts land in `out/`.
