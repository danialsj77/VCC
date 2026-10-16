"""Turn predicted mean expression (CPM per gene) into realistic integer-count cells.

Two generators:

* `TemplateGenerator`: edits randomly drawn *real control cells* of the same context (binomial thinning for
  decreased genes, added Poisson counts for increased genes). Most realistic: each cell keeps a real library size,
  zero pattern and spread. Open rules question: Arc calls the controls "model inputs only", so 03_make_submission
  refuses this generator unless --allow-control-templates is passed.
* `NBGenerator` (fallback): samples every count from a negative binomial fitted on the controls. It is
  depth-aware: each generated cell copies the library size and library-size decile of a randomly drawn control
  cell, and every gene's expected share follows how that gene's share changes with depth in the controls. No
  control cell's counts are copied.

Both take a predicted mean-CPM profile per target (the quantity the scorer's fold changes use), preserve it in
expectation, and guarantee the upload rules (non-negative integers, < 1e6 counts per cell).

Random numbers: both generators take the caller's `rng` and draw the *cells to imitate* first (template cells,
or the control cells whose depth is copied). The callers (`harness.write_prediction`, `submission.build_submission`)
give every target its own seeded generator, so two model variants are scored on the same template draws for
every target, whatever the other targets predict.
"""

import numpy as np
import scipy.sparse as sp

from .common import DE_GATE_CPM, mean_cpm

MAX_CELL_TOTAL = 900_000     # the upload rejects cells above 1,000,000 counts; real cells have ~20k
DEPTH_BINS = 10              # NBGenerator: control cells are split into this many library-size bins (deciles)


def _finish(out, rng, gated):
    """Shared safety steps: cap absurd cell totals; give every DE-testable gene at least one count."""
    tot = out.sum(axis=1)
    big = tot > MAX_CELL_TOTAL
    if big.any():
        out[big] = rng.binomial(out[big], (MAX_CELL_TOTAL / tot[big])[:, None])
    if gated is not None and gated.size:
        zero = gated[out[:, gated].sum(axis=0) == 0]
        out[rng.integers(0, out.shape[0], zero.size), zero] += 1
    return sp.csr_matrix(out.astype(np.float32))


def _normalize(target_cpm):
    t = np.clip(np.asarray(target_cpm, dtype=np.float64), 0, None)
    s = t.sum()
    if s <= 0:
        raise ValueError("predicted profile is all zeros")
    return t / s * 1e6


class TemplateGenerator:
    """Edit real control cells of one context so their mean expression follows the prediction."""

    def __init__(self, ctrl_X, spread=0.0):
        self.X = sp.csr_matrix(ctrl_X)
        self.ctrl_mean = mean_cpm(self.X)
        self.gated = np.flatnonzero(self.ctrl_mean > DE_GATE_CPM)
        self.spread = float(spread)

    def __call__(self, target_cpm, n_cells, rng):
        target = _normalize(target_cpm)
        # First draw: which control cells serve as templates. Under per-target seeding this draw is identical for
        # every model variant (it does not depend on the prediction), so variants edit the same cells.
        T = self.X[rng.integers(0, self.X.shape[0], n_cells)].toarray().astype(np.int64)
        lib = T.sum(axis=1, keepdims=True).astype(np.float64)
        expressed = self.ctrl_mean > 0
        ratio = np.ones_like(target)
        ratio[expressed] = target[expressed] / self.ctrl_mean[expressed]
        down, up = ratio < 1.0, ratio > 1.0

        # Down: keep each molecule with probability ratio (per-cell Beta draws with that mean if spread > 0).
        keep = np.ones(T.shape)
        if self.spread > 0:
            k = 1.0 / self.spread
            r = np.clip(ratio[down], 1e-6, 1 - 1e-6)
            keep[:, down] = rng.beta(k * r, k * (1 - r), size=(n_cells, down.sum()))
        else:
            keep[:, down] = ratio[down]
        out = rng.binomial(T, keep)

        # Up: add Poisson molecules, expected (ratio − 1) × this cell's expected control count.
        extra = np.zeros(T.shape)
        extra[:, up] = (ratio[up] - 1.0) * self.ctrl_mean[up] * lib / 1e6
        if self.spread > 0:
            extra[:, up] *= rng.gamma(1.0 / self.spread, self.spread, size=(n_cells, up.sum()))
        extra[:, ~expressed] = target[~expressed] * lib / 1e6          # genes induced from zero
        out = out + rng.poisson(extra)
        return _finish(out, rng, self.gated)


class NBGenerator:
    """Sample cells from a depth-aware negative binomial fitted on the controls (no control cell is copied).

    Why depth-aware. A gene's share of a cell's counts changes with the cell's library size (e.g. mitochondrial and
    ribosomal genes take a different share in deep vs shallow cells). The scorer's DE test averages per-cell CPM,
    but its PDS/MSE profile is the *ratio of summed counts* (pseudobulk), which weights deep cells more. A model with
    one share per gene (mu = cpm · lib / 1e6) matches the first and misses the second: on context A a "no change"
    prediction landed ~20-30% further (squared distance) from the control pseudobulk than real 400-cell groups do,
    a bias every target would carry. Tying each gene's share to the depth bin removes that bias: the distance is
    then within a few % of real groups' (tests/test_generator.py checks this).

    Model, for generated cell i and gene g:
        pick a control cell c(i) at random; lib_i = its library size, d_i = its library-size decile;
        mu_ig = target_cpm_g · H[d_i, g] · lib_i / 1e6
        y_ig ~ NB(mean mu_ig, dispersion phi_g)      (Gamma–Poisson mixture; Poisson where phi_g = 0)
    with
        H[d, g] = (mean per-cell CPM of g over control cells in decile d) / (mean per-cell CPM of g over all controls)
                  (1 where the overall mean is 0).
    Because deciles are drawn in proportion to their size, the average of H[d_i, g] over generated cells is 1, so
    the mean per-cell CPM (what the fold changes use) still follows `target_cpm` in expectation.

    phi_g: pooled method-of-moments estimate under the same depth-aware means,
        phi_g = sum_i [(y_ig − m_ig)^2 − m_ig] / sum_i m_ig^2,   m_ig = (decile-mean CPM of g) · lib_i / 1e6,
    clipped to [0, max_phi]. Fitting phi with the depth-aware means keeps the depth-driven differences in mean out
    of the dispersion (otherwise they would be counted twice: once in H, once as extra noise).
    """

    def __init__(self, ctrl_X, max_phi=50.0):
        X = sp.csr_matrix(ctrl_X, dtype=np.float64)
        n = X.shape[0]
        if n == 0:
            raise ValueError("NBGenerator needs at least one control cell")
        self.lib = np.asarray(X.sum(axis=1)).ravel()

        # ---- library-size deciles: equal-sized bins by rank (ties broken by cell order, so it's deterministic) ----
        n_bins = max(1, min(DEPTH_BINS, n))
        rank = np.empty(n, dtype=np.int64)
        rank[np.argsort(self.lib, kind="stable")] = np.arange(n)
        self.dec = rank * n_bins // n                                    # (n,) decile of each control cell
        self.n_bins = n_bins

        # ---- per-decile sums via weighted "indicator" matrices (bins × cells), so X is never copied densely ----
        # by_bin(w) is a (bins × cells) matrix with w_i in row dec_i, column i, so (by_bin(w) @ X)[d, g] =
        # sum over the cells of decile d of w_i · X[i, g]. With w = 1e6/lib that is the sum of per-cell CPM.
        cells = np.arange(n)
        with np.errstate(divide="ignore"):
            inv_lib = np.where(self.lib > 0, 1e6 / self.lib, 0.0)        # empty cells contribute 0 CPM (as mean_cpm)

        def by_bin(weights):
            return sp.csr_matrix((weights, (self.dec, cells)), shape=(n_bins, n))

        n_d = np.bincount(self.dec, minlength=n_bins).astype(np.float64)  # cells per decile
        cpm_sum = np.asarray((by_bin(inv_lib) @ X).todense())            # (bins, G) sum of per-cell CPM
        self.ctrl_mean = cpm_sum.sum(axis=0) / n                         # = mean_cpm(X): the scorer's group mean
        self.gated = np.flatnonzero(self.ctrl_mean > DE_GATE_CPM)
        dec_mean = cpm_sum / n_d[:, None]                                # (bins, G) mean per-cell CPM per decile
        self.H = np.ones_like(dec_mean)
        pos = self.ctrl_mean > 0
        self.H[:, pos] = dec_mean[:, pos] / self.ctrl_mean[pos]

        # ---- dispersion: pooled moments with the depth-aware means m_ig = c[d, g] · lib_i ----
        # sum_i (y − c·lib)^2 − c·lib  =  S_yy − 2c·S_yl + c²·L2 − c·L1   (sums over the cells of each decile)
        c = dec_mean / 1e6
        S_yy = np.asarray((by_bin(np.ones(n)) @ X.multiply(X)).todense())  # sum of y²
        S_yl = np.asarray((by_bin(self.lib) @ X).todense())                # sum of y · lib
        L1 = np.bincount(self.dec, weights=self.lib, minlength=n_bins)[:, None]
        L2 = np.bincount(self.dec, weights=self.lib ** 2, minlength=n_bins)[:, None]
        num = (S_yy - 2.0 * c * S_yl + c * c * L2 - c * L1).sum(axis=0)
        den = (c * c * L2).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = num / den                                              # 0/0 (gene never seen) → nan → 0
        self.phi = np.clip(np.nan_to_num(phi, nan=0.0, posinf=max_phi, neginf=0.0), 0.0, max_phi)

    def __call__(self, target_cpm, n_cells, rng):
        target = _normalize(target_cpm)
        # First draw: which control cells' depth (library size and decile) to imitate. It does not depend on the
        # prediction, so under per-target seeding every model variant gets the same depths for a given target.
        pick = rng.integers(0, self.lib.size, n_cells)
        lib = self.lib[pick][:, None]
        # mu_ig = target_g · H[d_i, g] · lib_i / 1e6: scale the (bins × G) table once, then pick each cell's row.
        mu = (self.H * (target / 1e6))[self.dec[pick]] * lib             # (n_cells, G) expected counts
        lam = mu.copy()
        nb = self.phi > 0
        shape = 1.0 / self.phi[nb]
        lam[:, nb] = rng.gamma(shape[None, :], mu[:, nb] / shape[None, :])   # Gamma–Poisson = negative binomial
        out = rng.poisson(lam)
        return _finish(out, rng, self.gated)
