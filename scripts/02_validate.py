"""Step 2 of model v1: choose the model settings on the H1 proxy with Arc's OFFICIAL scorer.

    python scripts/02_validate.py --config configs/v1.yaml [--gpu] [--smoke] [--resume]

The exam
--------
H1 is a real cell line with known answers that the model never sees as a source here: predictions use K562 ONLY
(METHOD.md: "don't train a model on H1 while evaluating on it"). For each variant:

1. Cross-fitting. Proxy targets are split into K folds (seeded). For fold f, the gains (beta, gamma) are fitted
   with `fit_gains` on the OTHER folds' H1 responses, then the fold's targets are predicted. So no target is
   scored with gains that saw its own answer. The template is always computed over the whole proxy panel (it uses
   only K562 data, so this leaks nothing), exactly as the submission computes it over the whole VCC panel.
2. The predicted mean CPM (relative to the proxy's control mean CPM) becomes 400 cells per target with the variant's
   generator (template or NB) built on the proxy's own control cells. The cells of target i are drawn from their
   own random stream (seed, i), so two variants that predict the same profile for a target get the same cells:
   differences between variants come from the predictions, not from which control cells happened to be drawn.
3. The official scorer (vccpipe.harness: cell-eval2, vcc2026 profile, scaled like the leaderboard) scores it.

A reference prediction, `control_paste` (every target = the control mean, i.e. "predict no change"), is scored once.

Two stages
----------
* Stage 1: the config's `validation.grid` (full factorial over z0, null_p, generator, spread) plus the ablations.
* Stage 2: for EACH generator, the best stage-1 grid variant of that generator is taken as the centre, and the
  config's `validation.stage2` grid (beta_scale x min_called_lfc) is applied on top of its settings. These knobs
  act after the gains are fitted (fit_gains never sees them): the least-squares gains shrink noisy transfers, but
  the scorer's DE metrics punish predicted changes too small to be called, so a larger effect than the "best fit"
  may score better. Running them around each winner rather than as a full factorial keeps the job short
  (5 extra scorings per generator instead of 6x more variants).

Choosing the winners
--------------------
Only grid and stage-2 variants can win (ablations are diagnostics; a warning is printed if one scores higher).
For each generator the best variant is written to best_params_<generator>.json, and the best overall to
best_params.json. The generator files matter because the template generator copies control cells (an open rules
question): 03_make_submission.py refuses a template-generator file unless told otherwise, and points to
best_params_nb.json. Gains are refitted on ALL proxy targets for every file written.

Outputs (in <work>/validation/)
-------------------------------
    results.csv                    one row per variant: settings, 7 scaled scores, raw metrics, fitted gains per fold
    best_params.json               the overall winner (see above), read by step 3 by default
    best_params_template.json      the best template-generator variant
    best_params_nb.json            the best NB-generator variant (no control cell copied)
    harness_cpu|gpu/               the scorer's bundle and per-run outputs (one per engine: the bundle depends on it)

Prediction files are deleted right after scoring (they are ~1 GB each at full size).

--resume keeps the rows of results.csv that were scored on THIS proxy (same proxy fingerprint and number of
targets) with THIS engine, and skips those variants (e.g. after a time-out). Rows from another engine or another
proxy are dropped (the original file is kept as results.<time>.csv). If the scorer's bundle had to be rebuilt
because the proxy changed, every old row is stale and results.csv is renamed even with --resume. Without --resume
an existing results.csv is always renamed to results.<time>.csv.
"""

import argparse
import hashlib
import itertools
import json
import sys
import time
import warnings
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vccpipe.cis import CisModel                                    # noqa: E402
from vccpipe.common import CONTROL, load_bundle_info, mean_cpm      # noqa: E402
from vccpipe.generator import NBGenerator, TemplateGenerator        # noqa: E402
from vccpipe.harness import ORDER, Harness, log_result, write_prediction  # noqa: E402
from vccpipe.model import EffectModel, ModelParams, fit_gains       # noqa: E402
from vccpipe.sources import ResponseSet                             # noqa: E402

VALIDATION_SOURCES = ("k562_gw",)          # never H1: it is the held-out line we score on
GENERATORS = ("template", "nb")
SELECTABLE = ("grid", "stage2")            # kinds of variant that may become best_params (not ablations)


# ---- config (the same small loader is in each script, so every script stands alone) -------------------------
def _merge(base, over):
    """Deep-merge `over` into `base`: dicts merge key by key, everything else is replaced."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path, smoke=False):
    """Read the YAML config; with smoke=True apply its `smoke:` section; make every path absolute (from ROOT)."""
    cfg = yaml.safe_load(Path(path).read_text())
    if smoke:
        cfg = _merge(cfg, cfg.get("smoke") or {})
    cfg["paths"] = {k: (ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return cfg


# ---- the variants ----------------------------------------------------------------------------------------------
def _fmt(v):
    return str(v).lower() if isinstance(v, bool) else f"{v:g}" if isinstance(v, float) else str(v)


def variant_params(cfg, variant):
    """ModelParams for a variant: config defaults + the variant's overrides.

    Validation can only test what it can load: the weights are limited to the validation sources (a weight for a
    source that is not loaded would only trigger a warning), and the template is the validation sources' template
    (template_sources = VALIDATION_SOURCES). That is also the template written into best_params*.json, so the
    submission keeps the template gamma was fitted for, whatever other sources it loads.
    """
    base = dict(cfg["model"])
    base["source_weights"] = {k: w for k, w in (base.get("source_weights") or {}).items()
                              if k in VALIDATION_SOURCES}
    base["template_sources"] = list(VALIDATION_SOURCES)
    return ModelParams.from_json({**base, **variant["model"]})


def _identity(cfg, variant):
    """What a variant actually computes: (generator, spread, the resolved ModelParams). Two variants with the same
    identity would score the same, so only the first one is kept."""
    return variant["generator"], float(variant["spread"]), variant_params(cfg, variant).to_json()


def expand_variants(cfg):
    """Stage 1: list of variants dict(name, kind, model={ModelParams overrides}, generator, spread, parent).

    The grid is a full factorial over its keys; keys that are not `generator`/`spread` must be ModelParams fields.
    The NB generator has no spread, so NB combinations differing only in spread are dropped as duplicates.
    Ablations are single overrides applied on top of the config's model/generator defaults.
    """
    model_fields = {f.name for f in fields(ModelParams)}
    gdef = cfg.get("generator", {})
    v = cfg["validation"]
    out, seen = [], set()

    def add(over, name, kind):
        over = dict(over)
        gen = over.pop("generator", gdef.get("name", "template"))
        spread = float(over.pop("spread", gdef.get("spread", 0.0)))
        if gen not in GENERATORS:
            raise ValueError(f"unknown generator {gen!r} (use one of {GENERATORS})")
        bad = sorted(set(over) - model_fields)
        if bad:
            raise ValueError(f"variant {name or over}: {bad} are not ModelParams fields")
        if gen == "nb":
            spread = 0.0
        key = (gen, spread, tuple(sorted((k, json.dumps(x)) for k, x in over.items())))
        if key in seen:
            return
        seen.add(key)
        if name is None:
            name = "_".join([f"{k}-{_fmt(x)}" for k, x in over.items()] + [gen] + ([f"s{_fmt(spread)}"]
                                                                              if gen == "template" else []))
        out.append({"name": name, "kind": kind, "model": over, "generator": gen, "spread": spread, "parent": ""})

    grid = v.get("grid") or {}
    keys = list(grid)
    for combo in itertools.product(*[list(grid[k]) for k in keys]):
        add(dict(zip(keys, combo)), None, "grid")
    for abl in v.get("ablations") or []:
        abl = dict(abl)
        name = abl.pop("name", None)
        add(abl, f"abl_{name}" if name else None, "ablation")
    names = [x["name"] for x in out]
    if len(set(names)) != len(names):
        raise ValueError(f"variant names are not unique: {names}")
    return out


def best_row(res, names, generator=None, kinds=SELECTABLE):
    """The highest-'overall' row of `res` among variants in `names` of the given kinds (and generator), or None.

    Ties keep the order of results.csv (stable sort), so the choice is deterministic.
    """
    c = res[res["variant"].isin(list(names)) & res["kind"].isin(list(kinds))]
    if generator is not None:
        c = c[c["generator"] == generator]
    c = c[np.isfinite(pd.to_numeric(c["overall"], errors="coerce"))]
    if c.empty:
        return None
    return c.sort_values("overall", ascending=False, kind="stable").iloc[0]


def stage2_variants(cfg, stage1, res):
    """Stage 2: around the best stage-1 GRID variant of each generator (by 'overall' in `res`), the full factorial
    of the config's validation.stage2 overrides. Variants that compute exactly the same as one already in
    `stage1` (e.g. beta_scale 1 and min_called_lfc 0, i.e. the winner itself) are skipped.

    Names: <winner>__<key>-<value>_<key>-<value>, e.g. z0-3.5_null_p-0.05_nb__beta_scale-2_min_called_lfc-0.4.
    """
    st2 = dict(cfg["validation"].get("stage2") or {})
    if not st2:
        return []
    model_fields = {f.name for f in fields(ModelParams)}
    bad = sorted(set(st2) - model_fields)
    if bad:
        raise ValueError(f"validation.stage2: {bad} are not ModelParams fields")
    keys = list(st2)
    seen = {_identity(cfg, v) for v in stage1}
    names = [v["name"] for v in stage1 if v["kind"] == "grid"]
    by_name = {v["name"]: v for v in stage1}
    out = []
    for gen in GENERATORS:
        row = best_row(res, names, generator=gen, kinds=("grid",))
        if row is None:
            continue                                   # this generator is not in the stage-1 grid
        win = by_name[row["variant"]]
        for combo in itertools.product(*[list(st2[k]) for k in keys]):
            over = dict(zip(keys, combo))
            v = {"name": f"{win['name']}__" + "_".join(f"{k}-{_fmt(x)}" for k, x in over.items()),
                 "kind": "stage2", "model": {**win["model"], **over}, "generator": gen, "spread": win["spread"],
                 "parent": win["name"]}
            ident = _identity(cfg, v)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(v)
    return out


# ---- helpers -------------------------------------------------------------------------------------------------
def load_proxy_controls(real_h5ad, genes):
    """The proxy's control cells (CSR counts), read in backed mode so the target cells never load."""
    a = ad.read_h5ad(real_h5ad, backed="r")
    if list(a.var_names) != list(genes):
        raise ValueError(f"{real_h5ad} is not on the VCC gene axis of the controls bundle")
    is_ctrl = (a.obs["target_gene"].astype(str) == CONTROL).to_numpy()
    X = sp.csr_matrix(a.X[is_ctrl], dtype=np.float32)       # boolean mask: contiguous runs are read as slices
    a.file.close()
    return X


def proxy_fingerprint(real_h5ad, targets_txt):
    """A short id of the proxy: hash of its target list + the size of real.h5ad. A rebuilt proxy with other
    settings (other targets, other cells) gets another id, so --resume never mixes rows from two proxies."""
    h = hashlib.sha1(Path(targets_txt).read_bytes()).hexdigest()[:10]
    return f"{h}-{Path(real_h5ad).stat().st_size}"


def make_folds(n, k, seed):
    """Fold index (0..k-1) of each of n targets: a seeded shuffle split into k near-equal parts."""
    k = max(2, min(int(k), n))
    fold = np.empty(n, dtype=np.int64)
    for f, idx in enumerate(np.array_split(np.random.default_rng(seed).permutation(n), k)):
        fold[idx] = f
    return fold, k


def cross_fit_predict(model, params, truth, targets, fold, ctrl_cpm):
    """Cross-fitted predicted mean CPM (T, G) and the gains fitted per fold.

    For fold f: gains from fit_gains on the other folds' H1 responses (truth restricted to those targets, the
    template still over the full panel), then the full panel is predicted and only fold f's rows are kept.
    The model's params are reset before every fit, so a degenerate fit (which returns the current gains) can
    never carry a previous fold's answer over. fit_gains estimates the RAW gains (it ignores beta_scale,
    gamma_scale and min_called_lfc); the prediction then applies the variant's scales on top of them.
    """
    pred = np.zeros((len(targets), ctrl_cpm.size), dtype=np.float64)
    gains = []
    for f in range(fold.max() + 1):
        train = [t for t, g in zip(targets, fold) if g != f]
        model.params = replace(params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            beta, gamma = fit_gains(model, truth.subset(train), targets=targets)
        for w in caught:
            print(f"      fold {f}: {w.message}")
        model.params = replace(params, beta=float(beta), gamma=float(gamma))
        test = np.flatnonzero(fold == f)
        pred[test] = model.predict_cpm(targets, ctrl_cpm)[test]
        gains.append((round(float(beta), 4), round(float(gamma), 4)))
    model.params = replace(params)
    return pred, gains


def result_columns():
    """The columns of results.csv, in order. Rows are appended, so every row must have exactly these."""
    return (["variant", "kind", "parent", "engine", "n_targets", "proxy_id", "folds", "generator", "spread"]
            + [f.name for f in fields(ModelParams)] + list(ORDER)
            + [f"raw_{m}" for m in ORDER if m != "overall"] + ["beta_mean", "gamma_mean", "gains_per_fold"])


def result_row(name, kind, parent, engine, n_targets, proxy_id, k, params, gen, spread, scaled, raw, gains):
    """One results.csv row (the columns of result_columns(), in that order)."""
    pdict = asdict(params) if params is not None else {}
    row = {"variant": name, "kind": kind, "parent": parent, "engine": engine, "n_targets": n_targets,
           "proxy_id": proxy_id, "folds": k, "generator": gen, "spread": spread}
    for f in fields(ModelParams):
        val = pdict.get(f.name, None)
        # dicts (source_weights) and lists (template_sources) go into one CSV cell as JSON
        row[f.name] = json.dumps(val, sort_keys=True) if isinstance(val, (dict, list)) else val
    row.update({m: float(scaled[m]) for m in ORDER})
    row.update({f"raw_{m}": float(raw[m]) for m in ORDER if m != "overall"})
    g = np.array(gains, dtype=float) if gains else np.full((1, 2), np.nan)
    row.update({"beta_mean": float(g[:, 0].mean()), "gamma_mean": float(g[:, 1].mean()),
                "gains_per_fold": json.dumps(gains)})
    assert list(row) == result_columns()
    return row


def comparable_rows(res, engine, n_targets, proxy_id):
    """Boolean mask of the rows of `res` that were scored with this engine on this proxy."""
    return ((res["engine"].astype(str) == engine)
            & (pd.to_numeric(res["n_targets"], errors="coerce") == n_targets)
            & (res["proxy_id"].astype(str) == proxy_id))


def prepare_results_csv(results_csv, resume, stale, engine, n_targets, proxy_id):
    """Decide what happens to an existing results.csv; returns the set of variants already scored (to skip).

    * no --resume, or the bundle was stale (scored against an old proxy): the file is renamed results.<time>.csv
    * --resume, but the file has other columns (written by another version of this script): renamed as well,
      because appending rows with other columns would corrupt it
    * --resume: rows from another engine or another proxy are dropped (the original is kept as a backup)
    """
    if not results_csv.exists():
        return set()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = results_csv.with_name(f"results.{stamp}.csv")
    n = 1
    while backup.exists():                       # never overwrite an earlier backup made in the same second
        backup = results_csv.with_name(f"results.{stamp}_{n}.csv")
        n += 1
    if not resume or stale:
        if resume and stale:
            print(f"  --resume: the scorer's bundle was rebuilt for a new proxy, so every scored row is stale; "
                  f"{results_csv.name} renamed to {backup.name}")
        results_csv.rename(backup)
        return set()
    header = list(pd.read_csv(results_csv, nrows=0).columns)
    if header != result_columns():
        print(f"  --resume: {results_csv.name} was written by another version of this script (other columns); "
              f"renamed to {backup.name}, starting a new one")
        results_csv.rename(backup)
        return set()
    res = pd.read_csv(results_csv)
    keep = comparable_rows(res, engine, n_targets, proxy_id)
    if not keep.all():
        print(f"  --resume: dropping {int((~keep).sum())} row(s) scored with another engine or on another proxy "
              f"(original kept as {backup.name})")
        results_csv.rename(backup)
        res[keep].to_csv(results_csv, index=False)
    return set(res.loc[keep, "variant"])


def edge_notes(cfg, params):
    """Grid keys (with >= 3 numeric values) at which the chosen params sit on the edge of the values tried."""
    notes = []
    grids = {**(cfg["validation"].get("grid") or {}), **(cfg["validation"].get("stage2") or {})}
    for k, vals in grids.items():
        if k in ("generator", "spread") or not hasattr(params, k):
            continue
        try:
            vals = sorted(float(x) for x in vals)
        except (TypeError, ValueError):
            continue
        if len(vals) >= 3 and float(getattr(params, k)) in (vals[0], vals[-1]):
            notes.append(f"{k} = {getattr(params, k):g} is at the edge of the values tried {vals}: the optimum may "
                         "lie outside the grid")
    return notes


# ---- main ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "v1.yaml"))
    ap.add_argument("--gpu", action="store_true", help="GPU scorer engine (gpudge), as on the platform")
    ap.add_argument("--smoke", action="store_true", help="apply the config's smoke section (tiny laptop test)")
    ap.add_argument("--resume", action="store_true", help="skip variants already scored on this proxy/engine")
    args = ap.parse_args()

    cfg = load_config(args.config, args.smoke)
    P, V = cfg["paths"], cfg["validation"]
    engine = "gpu" if args.gpu else "cpu"
    seed, cpp = int(V.get("seed", 0)), int(V.get("cells_per_pert", 400))
    proc, pdir = P["processed_dir"], P["work_dir"] / "h1_proxy"
    vdir = P["work_dir"] / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    real, tfile = pdir / "real.h5ad", pdir / "targets.txt"
    for f in [real, tfile, proc / "k562_gw.npz", proc / "h1_2025.npz", proc / "cis.json", proc / "tss.csv"]:
        if not f.exists():
            sys.exit(f"missing {f}: run scripts/01_build_sources.py first")

    # ---- inputs ----
    genes, _, _ = load_bundle_info(P["controls_dir"])
    k562 = ResponseSet.load(proc / "k562_gw.npz")
    truth = ResponseSet.load(proc / "h1_2025.npz")               # H1: the answers, used only to fit gains
    cis = CisModel.from_json((proc / "cis.json").read_text())
    tss = pd.read_csv(proc / "tss.csv")
    for rs in (k562, truth):
        if not np.array_equal(np.asarray(rs.genes).astype(str), genes):
            sys.exit(f"{rs.name} is not on the controls bundle's gene axis: rebuild with 01_build_sources.py")
    targets = tfile.read_text().split()
    # Every proxy target needs an H1 response (to fit gains), and the scorer needs every proxy target in the
    # prediction, so we cannot silently drop one. Both come from the same H1 files, so this only fails when the
    # proxy and h1_2025.npz were built from different inputs.
    missing = [t for t in targets if t not in set(map(str, truth.targets))]
    if missing:
        sys.exit(f"proxy targets without an H1 response: {missing[:10]} … rebuild both with 01_build_sources.py")
    ctrl_X = load_proxy_controls(real, genes)
    ctrl_cpm = mean_cpm(ctrl_X)
    fold, K = make_folds(len(targets), V.get("folds", 2), seed)
    proxy_id = proxy_fingerprint(real, tfile)
    in_k562 = sum(t in set(map(str, k562.targets)) for t in targets)
    # The strength filter of 01 (if the proxy was built with it): recorded in best_params*.json.
    sf_path = pdir / "strength_filter.json"
    strength = json.loads(sf_path.read_text()) if sf_path.exists() else None
    print(f"proxy {real} (id {proxy_id})\n  {len(targets)} targets ({in_k562} measured in K562), "
          f"{ctrl_X.shape[0]:,} control cells; {K} folds; scorer engine {engine}")
    if strength is None:
        print("  NOTE: no strength_filter.json next to the proxy: it was built without the strength filter")
    else:
        print(f"  strength filter of 01: {strength['n_excluded']} strong H1 targets were left out of the proxy")

    variants = expand_variants(cfg)
    st2 = cfg["validation"].get("stage2") or {}
    n_st2 = int(np.prod([len(x) for x in st2.values()])) if st2 else 0
    print(f"  stage 1: {len(variants)} variants + control_paste reference; stage 2: up to {max(n_st2 - 1, 0)} "
          f"variants around the stage-1 winner of each generator ({', '.join(st2) or 'none'})")

    # ---- scorer ----
    harness = Harness(real, vdir / f"harness_{engine}", gpu=args.gpu)
    t0 = time.time()
    print("building the scoring bundle (baseline + split-half replicates; once per proxy and engine) …", flush=True)
    # The harness reuses an existing bundle. If the proxy was rebuilt since (01 re-run), that bundle describes the
    # OLD proxy and every score would be silently wrong, so rebuild it.
    manifest = harness.bundle / "manifest.json"
    stale = manifest.exists() and manifest.stat().st_mtime < real.stat().st_mtime
    if stale:
        print("  the proxy is newer than the bundle: rebuilding the bundle")
    harness.build_bundle(force=stale)
    print(f"  bundle ready ({time.time() - t0:,.0f} s)")

    results_csv = vdir / "results.csv"
    done = prepare_results_csv(results_csv, args.resume, stale, engine, len(targets), proxy_id)

    generators = {}

    def generator(name, spread):
        """Generators are built once per (kind, spread) on the proxy's control cells and reused."""
        key = (name, float(spread))
        if key not in generators:
            generators[key] = TemplateGenerator(ctrl_X, spread) if name == "template" else NBGenerator(ctrl_X)
        return generators[key]

    def score(name, pred_cpm, gen, spread):
        path = vdir / f"pred_{name}.h5ad"
        try:
            write_prediction(path, targets, pred_cpm, generator(gen, spread), ctrl_X, genes, seed=seed,
                             cells_per_pert=cpp)
            return harness.score(name, path)                     # deletes the prediction file
        finally:
            path.unlink(missing_ok=True)                         # also when scoring failed

    def run(vlist, label):
        """Score every variant of vlist that is not in results.csv yet."""
        for i, v in enumerate(vlist, 1):
            if v["name"] in done:
                print(f"▷ {v['name']}: already in results.csv, skipping")
                continue
            t0 = time.time()
            print(f"▶ {label} [{i}/{len(vlist)}] {v['name']}", flush=True)
            params = variant_params(cfg, v)
            model = EffectModel([k562], params, cis=cis, tss=tss)
            pred, gains = cross_fit_predict(model, params, truth, targets, fold, ctrl_cpm)
            scaled, raw = score(v["name"], pred, v["generator"], v["spread"])
            log_result(results_csv, **result_row(v["name"], v["kind"], v["parent"], engine, len(targets), proxy_id,
                                                 K, params, v["generator"], v["spread"], scaled, raw, gains))
            done.add(v["name"])
            print(f"    overall {scaled['overall']:.3f}  gains (beta, gamma) per fold {gains}  "
                  f"({time.time() - t0:,.0f} s)", flush=True)

    def read_results():
        res = pd.read_csv(results_csv)
        return res[comparable_rows(res, engine, len(targets), proxy_id)]

    # ---- reference: "no change" for every target ----
    gdef = cfg.get("generator", {})
    ref_gen, ref_spread = gdef.get("name", "template"), float(gdef.get("spread", 0.0))
    if "control_paste" not in done:
        print("▶ control_paste (reference: control mean for every target)", flush=True)
        scaled, raw = score("control_paste", np.tile(ctrl_cpm, (len(targets), 1)), ref_gen, ref_spread)
        log_result(results_csv, **result_row("control_paste", "reference", "", engine, len(targets), proxy_id, K,
                                             None, ref_gen, ref_spread, scaled, raw, []))
        done.add("control_paste")
        print(f"    overall {scaled['overall']:.3f}")

    # ---- stage 1: the grid and the ablations ----
    run(variants, "stage 1")

    # ---- stage 2: beta_scale x min_called_lfc around each generator's stage-1 winner ----
    stage2 = stage2_variants(cfg, variants, read_results())
    if stage2:
        centres = sorted({v["parent"] for v in stage2})
        print(f"\nstage 2: {len(stage2)} variants around {', '.join(centres)}", flush=True)
        run(stage2, "stage 2")
    all_variants = variants + stage2
    by_name = {v["name"]: v for v in all_variants}

    # ---- choose per generator and overall, refit on all proxy targets, report ----
    res = read_results()
    names = [v["name"] for v in all_variants]
    ref = res[res["variant"] == "control_paste"]
    ref_scores = {m: float(ref.iloc[-1][m]) for m in ORDER} if len(ref) else None

    def write_best(path, row):
        """Refit the gains of the variant in `row` on all proxy targets and write its parameter file."""
        v = by_name[row["variant"]]
        params = variant_params(cfg, v)
        model = EffectModel([k562], params, cis=cis, tss=tss)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            beta, gamma = fit_gains(model, truth.subset(targets), targets=targets)
        for w in caught:
            print(f"  refit {v['name']}: {w.message}")
        # The submitted model: the refitted raw gains, the variant's scales and gates, the config's source weights
        # (v1: H1 weight 0), and the template of the validation sources only (what gamma was fitted for).
        final = replace(params, beta=float(beta), gamma=float(gamma), template_sources=list(VALIDATION_SOURCES),
                        source_weights={str(k): float(w)
                                        for k, w in (cfg["model"].get("source_weights") or {}).items()})
        out = {
            "variant": v["name"],
            "kind": v["kind"],
            "model": json.loads(final.to_json()),
            "generator": v["generator"],
            "spread": v["spread"],
            "fitted_gains": {"beta": float(beta), "gamma": float(gamma), "fitted_on": "h1_2025 (all proxy targets)",
                             "n_targets": len(targets),
                             "effective": {"beta": float(beta) * final.beta_scale,
                                           "gamma": float(gamma) * final.gamma_scale}},
            "cv_gains_per_fold": json.loads(row["gains_per_fold"]),
            "cv_scores": {m: float(row[m]) for m in ORDER},
            "control_paste_scores": ref_scores,
            "engine": engine,
            "validation_sources": list(VALIDATION_SOURCES),
            "proxy": {"path": str(real), "id": proxy_id, "n_targets": len(targets), "n_ctrl": int(ctrl_X.shape[0]),
                      "folds": K, "excluded_strong": None if strength is None else int(strength["n_excluded"]),
                      "strength_filter": strength},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(out, indent=2) + "\n")
        return final, beta, gamma

    best = best_row(res, names)
    if best is None:
        sys.exit("no scored grid or stage-2 variants in results.csv")
    written = {}
    for gen in GENERATORS:
        path = vdir / f"best_params_{gen}.json"
        row = best_row(res, names, generator=gen)
        if row is None:
            if path.exists():                    # an old file would look like a result of this run
                path.unlink()
                print(f"  removed {path.name} from an earlier run (no {gen} variant in this run)")
            continue
        written[gen] = (row, *write_best(path, row))
    final, beta, gamma = write_best(vdir / "best_params.json", best)

    show = ["variant", "kind", *ORDER, "beta_mean", "gamma_mean"]
    ranked = res.sort_values("overall", ascending=False, kind="stable")
    with pd.option_context("display.width", 180, "display.max_columns", 20, "display.max_colwidth", 60):
        print("\n=== Ranked by overall (scaled: 0 = context-mean baseline, 1 = replicate) ===")
        print(ranked[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    for gen, (row, p, b, g) in written.items():
        print(f"best {gen:8s}: {row['variant']}  overall {row['overall']:.3f}; gains refitted on all "
              f"{len(targets)} proxy targets: beta {b:.3f} x beta_scale {p.beta_scale:g}, gamma {g:.3f} x "
              f"gamma_scale {p.gamma_scale:g}, min_called_lfc {p.min_called_lfc:g}  → best_params_{gen}.json")
    print(f"best overall : {best['variant']} (generator {best['generator']}) → best_params.json")
    if best["generator"] == "template":
        print("  NOTE: the template generator copies the provided control cells (open rules question). "
              "03_make_submission.py uses it only with --allow-control-templates; otherwise pass "
              f"--params {vdir / 'best_params_nb.json'}")

    # ---- red flags ----
    if ref_scores is not None and best["overall"] <= ref_scores["overall"]:
        print("WARNING: the best variant does not beat control_paste on this proxy. Check the inputs before "
              "submitting anything.")
    abl = best_row(res, names, kinds=("ablation",))
    if abl is not None and abl["overall"] > best["overall"]:
        print(f"WARNING: the ablation {abl['variant']} ({abl['overall']:.3f}) beats every full variant "
              f"({best['overall']:.3f}): the part it switches off adds noise on this proxy.")
    for note in edge_notes(cfg, final):
        print(f"NOTE: {note}")
    print(f"wrote {', '.join(f'best_params_{g}.json' for g in written)} and best_params.json in {vdir}")
    leftovers = sorted(vdir.glob("pred_*.h5ad"))
    if leftovers:
        print(f"WARNING: prediction files left behind: {[p.name for p in leftovers]}")


if __name__ == "__main__":
    main()
