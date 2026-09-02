"""Tests for the hand-written matrix factorization in community_notes/mf.py."""
import numpy as np
import pytest

from community_notes import mf


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def test_gradient_matches_jax():
    """The hand-derived gradient agrees with autodiff of the same loss in jax."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(1)
    n_raters, n_notes, k, n_ratings = 7, 5, 2, 40
    u = rng.integers(0, n_raters, n_ratings)
    n = rng.integers(0, n_notes, n_ratings)
    r = rng.choice([0.0, 0.5, 1.0], n_ratings)
    p = mf.init_params(n_raters, n_notes, k, seed=3)
    p.mu = np.array(0.3)
    p.rater_intercept = rng.normal(size=n_raters)
    p.note_intercept = rng.normal(size=n_notes)

    def jax_loss(mu, bu, bn, fu, fn):
        pred = mu + bu[u] + bn[n] + jnp.sum(fu[u] * fn[n], axis=1)
        reg_i = jnp.mean(bu**2) + jnp.mean(bn**2) + mu**2
        reg_f = jnp.mean(fu**2) + jnp.mean(fn**2)
        return jnp.mean((pred - r) ** 2) + mf.LAMBDA_INTERCEPT * reg_i + mf.LAMBDA_FACTOR * reg_f

    assert np.isclose(mf.loss(p, u, n, r), float(jax_loss(*p.arrays())))
    expected = jax.grad(jax_loss, argnums=(0, 1, 2, 3, 4))(*p.arrays())
    for hand, auto in zip(mf.gradient(p, u, n, r).arrays(), expected):
        auto = np.asarray(auto)
        assert np.linalg.norm(hand - auto) <= 1e-5 * np.linalg.norm(auto)


def planted_population(seed=0, n_raters=200, n_notes=100, density=0.3, noise=0.05):
    """Ratings sampled from a known model, so the fit can be checked against truth."""
    rng = np.random.default_rng(seed)
    truth = mf.Params(
        mu=np.array(0.3),
        rater_intercept=rng.normal(0, 0.1, n_raters),
        note_intercept=rng.normal(0, 0.3, n_notes),
        rater_factor=rng.normal(0, 1.0, (n_raters, 1)),
        note_factor=rng.normal(0, 0.5, (n_notes, 1)),
    )
    mask = rng.random((n_raters, n_notes)) < density
    u, n = np.nonzero(mask)
    r = mf.predict(truth, u, n) + rng.normal(0, noise, len(u))
    data = dict(
        rater_idx=u.astype(np.int32),
        note_idx=n.astype(np.int32),
        rating=r.astype(np.float32),
        rater_ids=np.arange(n_raters),
        note_ids=np.arange(n_notes),
    )
    return data, truth


def test_recovers_planted_parameters():
    data, truth = planted_population()
    p, _ = mf.fit(data, seed=0, max_iters=500)
    assert pearson(p.note_intercept, truth.note_intercept) > 0.9
    assert abs(pearson(p.note_factor[:, 0], truth.note_factor[:, 0])) > 0.9


def two_camp_population(seed=0, n_per_camp=20):
    """Two rater camps at f = +1 and f = -1 rating a mix of notes.

    Ten notes are liked only by the + camp, ten only by the - camp, eight are
    disliked by everyone and two are liked by everyone. Returns the data and
    a kind label per note.
    """
    rng = np.random.default_rng(seed)
    camp = np.r_[np.ones(n_per_camp), -np.ones(n_per_camp)]
    kinds = ["plus_only"] * 10 + ["minus_only"] * 10 + ["bad"] * 8 + ["good"] * 2
    rows = []
    for j, kind in enumerate(kinds):
        for i in range(len(camp)):
            if kind == "plus_only":
                val = 1.0 if camp[i] > 0 else 0.0
            elif kind == "minus_only":
                val = 1.0 if camp[i] < 0 else 0.0
            else:
                val = 1.0 if kind == "good" else 0.0
            if rng.random() < 0.1:
                val = 1.0 - val
            rows.append((i, j, val))
    a = np.array(rows)
    data = dict(
        rater_idx=a[:, 0].astype(np.int32),
        note_idx=a[:, 1].astype(np.int32),
        rating=a[:, 2].astype(np.float32),
        rater_ids=np.arange(len(camp)),
        note_ids=np.arange(len(kinds)),
    )
    return data, kinds


def test_bridging_is_what_the_intercept_measures():
    """A note both camps call helpful should clear the threshold. A note only
    one camp likes is explained by the factor term, so its intercept stays low
    however many ratings it has. This is the property the algorithm exists for.
    """
    data, kinds = two_camp_population()
    p, _ = mf.fit(data, seed=0, max_iters=1000)
    status = mf.note_status(p, mf.ratings_per_note(data))
    note_a = kinds.index("good")
    note_b = kinds.index("plus_only")
    assert status[note_a] == mf.CURRENTLY_RATED_HELPFUL
    assert status[note_b] != mf.CURRENTLY_RATED_HELPFUL
    assert p.note_intercept[note_a] > p.note_intercept[note_b]
    assert np.sign(p.rater_factor[0, 0]) != np.sign(p.rater_factor[-1, 0])


def test_loss_decreases_for_both_optimizers():
    data, _ = planted_population(seed=2, n_raters=60, n_notes=40)
    _, adam = mf.fit(data, optimizer="adam", seed=0, max_iters=300)
    _, gd = mf.fit(data, optimizer="gd", seed=0, max_iters=1500)
    assert np.all(np.diff(gd) <= 1e-12)
    assert gd[-1] < 0.5 * gd[0]
    assert adam[-1] < 0.5 * adam[0]


def test_note_status_thresholds():
    p = mf.Params(
        mu=np.array(0.0),
        rater_intercept=np.zeros(1),
        note_intercept=np.array([0.40, 0.39, -0.13, -0.12, 0.50, 0.50]),
        rater_factor=np.zeros((1, 1)),
        note_factor=np.array([[0.0], [0.0], [0.1], [0.1], [0.6], [0.0]]),
    )
    num_ratings = np.array([5, 5, 5, 5, 5, 4])
    status = mf.note_status(p, num_ratings)
    assert list(status) == [
        mf.CURRENTLY_RATED_HELPFUL,  # right at 0.40
        mf.NEEDS_MORE_RATINGS,  # just under
        mf.CURRENTLY_RATED_NOT_HELPFUL,  # right at -0.05 - 0.8 * 0.1
        mf.NEEDS_MORE_RATINGS,  # just above that line
        mf.NEEDS_MORE_RATINGS,  # high intercept but |factor| > 0.5
        mf.NEEDS_MORE_RATINGS,  # fewer than 5 ratings
    ]
