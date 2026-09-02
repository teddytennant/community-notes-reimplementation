# The Community Notes scoring algorithm, as reimplemented here

This note records what the official algorithm does, where each number came
from, and what this project leaves out. Everything was read from the public
scorer at https://github.com/twitter/communitynotes at commit 2704b2d
(20 August 2026). File paths below are relative to `scoring/src/scoring/`
in that repo. The "guide" is the Under the Hood page
https://communitynotes.x.com/guide/en/under-the-hood/ranking-notes, which the
repo mirrors at `documentation/under-the-hood/ranking-notes.md`; I read the
mirror.

## Ratings

A rater answers "Is this note helpful?" with yes, somewhat or no. The guide
maps these to 1.0, 0.5 and 0.0, and `process_data.py` does the same when it
builds the `helpfulNum` column (older two-option ratings map to 1 and 0).
That column is the only thing the model sees.

## The model

Each rating by rater u of note n is predicted as

    pred(u, n) = mu + b_u + b_n + f_u . f_n

where mu is a global intercept, b_u and b_n are rater and note intercepts,
and f_u and f_n are factor vectors. `matrix_factorization/model.py` is the
whole model: four embedding tables plus one scalar, added up in `forward`.
The production model uses a single factor dimension (`numFactors=1` in
`MatrixFactorization.__init__`, and the guide says they keep it at 1 to avoid
overfitting). The note's score is b_n. Nothing else about the note goes into
its status apart from the size of f_n.

The idea is in the guide's Matrix Factorization section: the factor term can
explain a rating by "this rater and this note lean the same way", and it's
cheap for the model to use because factors are lightly regularized. The
intercept is expensive. So b_n only grows when raters on both sides of the
factor axis rate the note helpful, which is the bridging property.

## The loss

`MatrixFactorization._get_loss` and `_get_reg_loss` compute

    L = mean over ratings of (pred - r)^2
      + 0.15 * (mean(b_u^2) + mean(b_n^2) + mu^2)
      + 0.03 * (mean(f_u^2) + mean(f_n^2))

with the lambdas set in `MatrixFactorization.__init__`: `userFactorLambda`
and `noteFactorLambda` are 0.03, and `userInterceptLambda`,
`noteInterceptLambda` and `globalInterceptLambda` are all `0.03 * 5`. The
guide states the same numbers as lambda_i = 0.15 and lambda_f = 0.03.

One thing to notice: the guide writes the loss as sums, but the code averages
the squared error over ratings and averages each parameter block's squares
before multiplying by its lambda. This project follows the code. The
difference matters for how strong the penalty feels per parameter, since a
mean over notes makes each note's share of the penalty 1/n_notes as large.

The same class also supports a sigmoid cross entropy loss, a "diamond"
penalty on |b_n * f_n|, and a rating-count-normalized version of the
penalty. All are off by default and this project skips them.

## Training

`_create_mf_model` uses `torch.optim.Adam` at learning rate 1.0 when
training from scratch (`noInitLearningRate`) and 0.2 when warm-started from
a previous fit (`initLearningRate`). `model.py` initializes intercepts to
zero and factors with `xavier_uniform_`, which for an (n, k) table means
uniform on (-a, a) with a = sqrt(6 / (n + k)). `_fit_model` runs full-batch:
every step computes the loss on all ratings, clips the gradient's global
norm to 1.0 (`clip_grad_norm_`), and steps. It stops when the loss changes
by less than `convergence = 1e-7`, or when the loss goes up after epoch 100.
`_flip_factors_for_identification` then flips the sign of the factor axis
if fewer than half the raters ended up negative, so different runs agree on
which side is which.

This project implements exactly that loop in numpy: `community_notes/mf.py`
has the loss, the hand-derived gradient, Adam, gradient clipping, the two
stopping rules and the sign flip, with the same constants. Plain gradient
descent is there too for comparison.

## Note status

A note starts as Needs More Ratings and can only leave that state with at
least 5 ratings (`minRatingsNeeded=5` in `mf_base_scorer.py`). The rules
that decide the rest live in `note_ratings.py`:

`is_crh`: Currently Rated Helpful when b_n >= 0.40 (`crhThreshold`).

`is_crnh_diamond`: Currently Rated Not Helpful when
b_n <= -0.05 + (-0.8) * |f_n| (`crnhThresholdIntercept` and
`crnhThresholdNoteFactorMultiplier`). The guide phrases it as
"less than -0.05 - 0.8 * abs(noteFactorScore)". A note with a large factor
needs an even lower intercept to be called Not Helpful, because a large
factor already says its ratings were one-sided.

A large-factor filter (`factorThreshold=0.5` in `mf_base_scorer.py`,
applied in `scoring_rules.py`) drops a note back to Needs More Ratings if it
would be Helpful but |f_n| > 0.5. The guide says Helpful notes must have
abs(f_n) < 0.50.

`community_notes/mf.py` keeps these as module constants and `note_status` applies
them in that order.

## What the official pipeline does that this project does not

The production core model is not one fit. `mf_base_scorer.py` runs the
factorization on ratings from one stable modeling group first, uses that to
initialize a fit on everything, computes rater helpfulness scores from the
result, drops ratings from raters below a helpfulness bar, fits again warm
started from the first fit, and in the final scoring pass fits once more with
rater parameters frozen. It also filters the input to raters with at least
10 ratings and notes with at least 5 raters before any of that. This project
fits once, from scratch, on whatever the data layer hands it. The tests show
that one round already has the bridging property; the extra rounds are about
robustness to manipulation and drift.

Around the thresholds above the production scorer stacks more rules: a
pseudo-rater sensitivity analysis that makes a note Not Helpful if the upper
bound on its intercept is under -0.04, a lower Not Helpful bar (-0.15) for
notes that say a post is not misleading, a 0.01 hysteresis so a note has to
fall below 0.39 to lose Helpful, tag-based filters that raise the Helpful
bar to 0.5 for notes with many "incorrect" or "sources missing" tags, net
helpful minimums from each side of the factor axis, baseline intercepts that
re-fit without high-volume or correlated raters, topic and group models, and
a refit when the training loss lands above 0.09. None of that is here. Each
one is a guard on top of the same intercept, and the goal of this project is
the intercept.
