# community-notes-reimplementation

A from-scratch reimplementation of the matrix factorization that decides which
Community Notes get shown on X. The model is the one in the public scorer:
each rating is predicted as a global mean plus a rater intercept, a note
intercept and a one dimensional rater factor times note factor. A note is
rated helpful when its intercept clears a threshold, which rewards notes that
raters on both sides of the factor axis liked, not notes that one side liked a
lot. Everything here is numpy with the gradients and Adam written out by hand.

## What came out of it

On two weeks of June 2026 ratings this agrees with the official core model on
89% of notes, and when it calls a note Helpful it agrees 73% of the time. It
finds only 27% of the notes the real scorer marks Helpful, which is what the
missing rounds of rater filtering buy them.

Fitting the same ratings under ten seeds moves 2.9% of statuses. 1,066 notes
are Helpful under some seed and only 598 under all ten. Full numbers in
`experiments/README.md`.

## Install

    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python -e '.[dev]'

## Run

Build a slice of the public data (see `data/README.md` for the window used):

    .venv/bin/python scripts/build_slice.py

Fit and compare against the official scores:

    .venv/bin/python scripts/fit.py --seed 0

Tests:

    .venv/bin/pytest

## Layout

`community_notes/data.py` loads the public dump and turns it into index arrays.
`community_notes/mf.py` is the model, loss, gradients and optimizer.
`docs/algorithm.md` records the constants and where each came from.
`experiments/` holds the scripts and outputs for the writeup.
