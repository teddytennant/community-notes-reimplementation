"""Community Notes matrix factorization, written out by hand in numpy.

    pred(u, n) = mu + b_u + b_n + f_u . f_n

Intercepts are regularized five times harder than factors, so the model would
rather explain a rating through viewpoint agreement than by lifting b_n. That
is what makes b_n a bridging score. Constants come from the open source scorer
at https://github.com/twitter/communitynotes; docs/algorithm.md says where each
one lives and what this file leaves out.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

LAMBDA_FACTOR = 0.03
LAMBDA_INTERCEPT = 0.03 * 5

ADAM_LR = 1.0
GD_LR = 0.5
GRAD_CLIP_NORM = 1.0
CONVERGENCE = 1e-7

CRH_THRESHOLD = 0.40
CRNH_INTERCEPT = -0.05
CRNH_FACTOR_MULTIPLIER = -0.8
CRH_MAX_ABS_FACTOR = 0.5
MIN_RATINGS = 5

CURRENTLY_RATED_HELPFUL = "CURRENTLY_RATED_HELPFUL"
CURRENTLY_RATED_NOT_HELPFUL = "CURRENTLY_RATED_NOT_HELPFUL"
NEEDS_MORE_RATINGS = "NEEDS_MORE_RATINGS"


@dataclass
class Params:
    """All model parameters. Intercepts are (count,), factors are (count, k)."""

    mu: np.ndarray
    rater_intercept: np.ndarray
    note_intercept: np.ndarray
    rater_factor: np.ndarray
    note_factor: np.ndarray

    def arrays(self) -> list[np.ndarray]:
        return [getattr(self, f.name) for f in fields(self)]


def init_params(n_raters: int, n_notes: int, k: int = 1, seed: int = 0) -> Params:
    """Zero intercepts and xavier-uniform factors, like the official model.py."""
    rng = np.random.default_rng(seed)

    def xavier(n):
        a = np.sqrt(6.0 / (n + k))
        return rng.uniform(-a, a, size=(n, k))

    return Params(
        mu=np.zeros(()),
        rater_intercept=np.zeros(n_raters),
        note_intercept=np.zeros(n_notes),
        rater_factor=xavier(n_raters),
        note_factor=xavier(n_notes),
    )


def predict(p: Params, u: np.ndarray, n: np.ndarray) -> np.ndarray:
    """mu + b_u + b_n + f_u . f_n for each (rater, note) pair."""
    dot = np.sum(p.rater_factor[u] * p.note_factor[n], axis=1)
    return p.mu + p.rater_intercept[u] + p.note_intercept[n] + dot


def _mean_sq(x: np.ndarray) -> float:
    return float(np.mean(x**2))


def loss(p, u, n, r, lam_f=LAMBDA_FACTOR, lam_i=LAMBDA_INTERCEPT) -> float:
    """Mean squared error, plus each parameter block's mean square times its lambda."""
    err = predict(p, u, n) - r
    reg_i = _mean_sq(p.rater_intercept) + _mean_sq(p.note_intercept) + float(p.mu**2)
    reg_f = _mean_sq(p.rater_factor) + _mean_sq(p.note_factor)
    return float(np.mean(err**2)) + lam_i * reg_i + lam_f * reg_f


def _scatter_add(idx: np.ndarray, rows: np.ndarray, size: int) -> np.ndarray:
    """Sum the (R, k) rows into `size` slots keyed by idx, the transpose of table[idx]."""
    out = np.empty((size, rows.shape[1]))
    for j in range(rows.shape[1]):
        out[:, j] = np.bincount(idx, weights=rows[:, j], minlength=size)
    return out


def gradient(p, u, n, r, lam_f=LAMBDA_FACTOR, lam_i=LAMBDA_INTERCEPT) -> Params:
    """Hand-derived gradient of `loss`, with g = 2 (pred - r) / R scattered back
    onto whichever parameter each rating touched."""
    n_raters, n_notes = len(p.rater_intercept), len(p.note_intercept)
    g = 2.0 * (predict(p, u, n) - r) / len(r)
    return Params(
        mu=np.sum(g) + 2 * lam_i * p.mu,
        rater_intercept=np.bincount(u, weights=g, minlength=n_raters)
        + 2 * lam_i * p.rater_intercept / n_raters,
        note_intercept=np.bincount(n, weights=g, minlength=n_notes)
        + 2 * lam_i * p.note_intercept / n_notes,
        rater_factor=_scatter_add(u, g[:, None] * p.note_factor[n], n_raters)
        + 2 * lam_f * p.rater_factor / p.rater_factor.size,
        note_factor=_scatter_add(n, g[:, None] * p.rater_factor[u], n_notes)
        + 2 * lam_f * p.note_factor / p.note_factor.size,
    )


def clip_by_global_norm(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    """Scale all gradients together so their joint L2 norm is at most max_norm."""
    total = np.sqrt(sum(np.sum(g**2) for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))
    return [g * scale for g in grads]


class GradientDescent:
    """x <- x - lr * g."""

    def __init__(self, lr: float = GD_LR):
        self.lr = lr

    def step(self, xs, gs):
        return [x - self.lr * g for x, g in zip(xs, gs)]


class Adam:
    """Adam (Kingma and Ba, 2015), matching torch.optim.Adam."""

    def __init__(self, lr: float = ADAM_LR, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, xs, gs):
        if self.m is None:
            self.m = [np.zeros_like(x) for x in xs]
            self.v = [np.zeros_like(x) for x in xs]
        self.t += 1
        out = []
        for i, (x, g) in enumerate(zip(xs, gs)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            out.append(x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return out


def flip_factors_for_identification(p: Params) -> Params:
    """Pick the sign of each factor axis so at least half the raters are negative.

    f_u . f_n is unchanged when both sides flip, so without this runs with
    different seeds would not line up.
    """
    rater_factor, note_factor = p.rater_factor.copy(), p.note_factor.copy()
    for j in range(rater_factor.shape[1]):
        f = rater_factor[:, j]
        nonzero = np.count_nonzero(f)
        if nonzero and np.sum(f < 0) / nonzero < 0.5:
            rater_factor[:, j] *= -1
            note_factor[:, j] *= -1
    return Params(p.mu, p.rater_intercept, p.note_intercept, rater_factor, note_factor)


def fit(
    data: dict,
    k: int = 1,
    seed: int = 0,
    optimizer: str = "adam",
    lr: float | None = None,
    lam_f: float = LAMBDA_FACTOR,
    lam_i: float = LAMBDA_INTERCEPT,
    tol: float = CONVERGENCE,
    max_iters: int = 2000,
    clip_norm: float | None = GRAD_CLIP_NORM,
    init: Params | None = None,
    flip: bool = True,
) -> tuple[Params, np.ndarray]:
    """Full-batch fit. Returns the parameters and the loss after every step.

    Stops on tol, on max_iters, or, like the official loop, when the loss
    rises after step 100.
    """
    u = np.asarray(data["rater_idx"], dtype=np.int64)
    n = np.asarray(data["note_idx"], dtype=np.int64)
    r = np.asarray(data["rating"], dtype=np.float64)
    n_raters = len(data["rater_ids"]) if "rater_ids" in data else int(u.max()) + 1
    n_notes = len(data["note_ids"]) if "note_ids" in data else int(n.max()) + 1

    p = init if init is not None else init_params(n_raters, n_notes, k, seed)
    if optimizer == "adam":
        opt = Adam(ADAM_LR if lr is None else lr)
    elif optimizer == "gd":
        opt = GradientDescent(GD_LR if lr is None else lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}, use 'adam' or 'gd'")

    history = [loss(p, u, n, r, lam_f, lam_i)]
    for it in range(max_iters):
        grads = gradient(p, u, n, r, lam_f, lam_i).arrays()
        if clip_norm is not None:
            grads = clip_by_global_norm(grads, clip_norm)
        p = Params(*opt.step(p.arrays(), grads))
        history.append(loss(p, u, n, r, lam_f, lam_i))
        if abs(history[-1] - history[-2]) < tol:
            break
        if it > 100 and history[-1] > history[-2]:
            break
    if flip:
        p = flip_factors_for_identification(p)
    return p, np.asarray(history)


def ratings_per_note(data: dict, n_notes: int | None = None) -> np.ndarray:
    """How many ratings each note received."""
    n = np.asarray(data["note_idx"], dtype=np.int64)
    if n_notes is None:
        n_notes = len(data["note_ids"]) if "note_ids" in data else int(n.max()) + 1
    return np.bincount(n, minlength=n_notes)


def note_status(p: Params, num_ratings: np.ndarray | None = None, min_ratings: int = MIN_RATINGS):
    """One status string per note from the official thresholds.

    Pass num_ratings=None to skip the minimum-ratings gate.
    """
    b = p.note_intercept
    f = np.linalg.norm(p.note_factor, axis=1)
    enough = np.ones(len(b), dtype=bool) if num_ratings is None else num_ratings >= min_ratings
    crh = enough & (b >= CRH_THRESHOLD) & (f <= CRH_MAX_ABS_FACTOR)
    crnh = enough & (b <= CRNH_INTERCEPT + CRNH_FACTOR_MULTIPLIER * f)
    status = np.full(len(b), NEEDS_MORE_RATINGS, dtype=object)
    status[crh] = CURRENTLY_RATED_HELPFUL
    status[crnh] = CURRENTLY_RATED_NOT_HELPFUL
    return status
