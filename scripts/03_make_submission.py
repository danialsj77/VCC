"""Step 3 of model v1: build the VCC submission with the settings chosen in step 2, check it, package it.

    python scripts/03_make_submission.py --config configs/v1.yaml --controls-dir controls \
        [--params work/validation/best_params_nb.json] [--allow-control-templates] \
        [--out work/submission_controls/submission.h5ad] [--dry-run] [--keep-h5ad] [--smoke]

What it does
------------
1. Effect model = the sources whose weight in the config's model.source_weights is > 0 (v1: K562 only; H1 has
   weight 0, because it is the held-out line the model was validated and its gains fitted on), the gains and
   settings of the parameter file, and the cis model. The template is built only from the parameter file's
   `template_sources` (["k562_gw"]): gamma was fitted for that template and means nothing for another one.
2. For every context of the bundle (by name, from manifest.json): predicted mean CPM for all panel targets
   relative to THAT context's control mean CPM, then cells from the chosen generator built on that context's
   controls (vccpipe.submission.build_submission writes them in chunks, so memory stays bounded). Before the cells
   are written, a sanity line per context: how many DE-testable genes (control CPM > 5) the template alone moves
   by more than 0.2 log2 for EVERY target (should be a handful; hundreds would put calls on every target).
3. `vcc prep --dry-run` checks the file exactly as the platform will.
4. Unless --dry-run (or --smoke): `vcc prep` packages it to <out>.vcc next to the h5ad, writes <out>.vcc.sha256,
   and deletes the ~17 GB h5ad (keep it with --keep-h5ad).

The generator. best_params.json holds the overall winner of step 2, which may use the TEMPLATE generator. That
generator edits copies of the provided control cells, and the data page says controls are "model inputs only — do
not copy them into your prediction" (open rules question). So this script refuses a template-generator parameter
file unless --allow-control-templates is passed; the safe choice is --params work/validation/best_params_nb.json
(NB generator: every count is sampled, no control cell is copied).

This script NEVER submits. It prints a neutral `vcc submit` command for the team lead, who submits after review
(2 scored submissions per day; in the final phase only the last entry counts). The leaderboard shows the model name
to everyone, so the name carries no method details; those stay in <out>.json and submissions/LOG.md.

FINAL PHASE: pass the D/E/F bundle with --controls-dir controls_DEF. If its gene axis differs from the one the
sources were built on (order, or a few genes added/dropped), the sources are mapped onto it by gene symbol
(ResponseSet.reindex_genes; genes a source lacks count as not measured) and a summary is printed.
--smoke: 5 cells per target (also passed to vcc prep as --cells-per-pert 5) and always a dry run.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vccpipe.cis import CisModel                                     # noqa: E402
from vccpipe.common import CELLS_PER_PERT, DE_GATE_CPM, load_bundle_info, mean_cpm  # noqa: E402
from vccpipe.generator import NBGenerator, TemplateGenerator         # noqa: E402
from vccpipe.model import EffectModel, ModelParams                   # noqa: E402
from vccpipe.sources import ResponseSet                              # noqa: E402
from vccpipe.submission import build_submission, vcc_prep            # noqa: E402

SUBMISSION_SOURCES = ("k562_gw", "h1_2025")     # candidates; a source is loaded only if its config weight is > 0
TEMPLATE_CHECK_LFC = 0.2                        # sanity check: |gamma_eff * T| above this on a DE-testable gene ...
TEMPLATE_CHECK_WARN = 50                        # ... on more than this many genes in a context → warning


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


def to_bundle_axis(rs, genes, where):
    """`rs` on the bundle's gene axis: unchanged if it already is, else mapped by symbol (with a printed summary)."""
    old = np.asarray(rs.genes).astype(str)
    if np.array_equal(old, genes):
        return rs
    old_set, new_set = set(old), set(genes)
    new_genes = [g for g in genes if g not in old_set]                 # bundle genes the source never saw
    dropped = [g for g in old if g not in new_set]                     # source genes the bundle does not score
    measured = set(old[np.asarray(rs.measured, dtype=bool)])
    print(f"  {rs.name}: built on another gene axis ({old.size:,} genes) than {where} ({genes.size:,} genes); "
          f"mapping it by gene symbol:\n"
          f"    {len(new_genes):,} bundle genes are new to it (treated as not measured)"
          f"{': ' + ', '.join(new_genes[:8]) + (' …' if len(new_genes) > 8 else '') if new_genes else ''}\n"
          f"    {len(dropped):,} of its genes are not in the bundle and are dropped "
          f"({sum(g in measured for g in dropped):,} of them measured)"
          f"{': ' + ', '.join(dropped[:8]) + (' …' if len(dropped) > 8 else '') if dropped else ''}\n"
          f"    {'same genes, other order' if not new_genes and not dropped else 'gene sets differ'}", flush=True)
    return rs.reindex_genes(genes)


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "v1.yaml"))
    ap.add_argument("--controls-dir", default=None, help="controls bundle (default: the config's controls_dir)")
    ap.add_argument("--params", default=None, help="default: <work>/validation/best_params_nb.json (the NB-generator "
                    "winner); with --allow-control-templates the default is best_params.json (the overall winner)")
    ap.add_argument("--allow-control-templates", action="store_true",
                    help="allow the template generator, which copies the provided control cells (open rules "
                         "question: use only once the organizers have confirmed it)")
    ap.add_argument("--out", default=None, help="default: <work>/submission_<controls folder name>/submission.h5ad")
    ap.add_argument("--dry-run", action="store_true", help="build and check only; do not package a .vcc")
    ap.add_argument("--keep-h5ad", action="store_true", help="keep the .h5ad after packaging (it is ~17 GB)")
    ap.add_argument("--smoke", action="store_true", help="config smoke section: 5 cells per target, dry run")
    args = ap.parse_args()

    cfg = load_config(args.config, args.smoke)
    P, S = cfg["paths"], cfg["submission"]
    dry_run = args.dry_run or args.smoke
    controls_dir = Path(args.controls_dir).resolve() if args.controls_dir else P["controls_dir"]
    # Default: the NB-generator winner (no control cell is copied). Only when the team has the organizers' written OK
    # for control-cell templates (--allow-control-templates) does the default become the overall winner.
    default_params = "best_params.json" if args.allow_control_templates else "best_params_nb.json"
    params_path = (Path(args.params) if args.params else P["work_dir"] / "validation" / default_params).resolve()
    # One output folder per controls bundle (work/submission_controls, work/submission_controls_DEF, ...), so the
    # final-phase file can never overwrite the validation-phase one.
    out_h5ad = (Path(args.out) if args.out
                else P["work_dir"] / f"submission_{Path(controls_dir).name}" / "submission.h5ad")
    out_h5ad = out_h5ad.resolve()
    cpp = int(S.get("cells_per_pert", CELLS_PER_PERT))
    if cpp != CELLS_PER_PERT and not dry_run:
        sys.exit(f"cells_per_pert = {cpp}: only a dry run is allowed with a non-standard cell count")

    # ---- settings chosen by validation (checked before anything heavy is loaded) ----
    if not params_path.exists():
        sys.exit(f"missing {params_path}: run scripts/02_validate.py first (or pass --params)")
    best = json.loads(params_path.read_text())
    gen_name, spread = best["generator"], float(best.get("spread", 0.0))
    if gen_name not in ("template", "nb"):
        sys.exit(f"unknown generator {gen_name!r} in {params_path}")
    if gen_name == "template" and not args.allow_control_templates:
        nb = params_path.parent / "best_params_nb.json"
        sys.exit(f"{params_path.name} uses the TEMPLATE generator, which copies the provided control cells into the "
                 "prediction. The data page says controls are 'model inputs only — do not copy them into your "
                 "prediction', and the organizers have not answered whether editing copies is allowed.\n"
                 f"  Use the NB generator instead:  --params {nb}"
                 f"{'' if nb.exists() else '  (missing: rerun scripts/02_validate.py to write it)'}\n"
                 "  or, only once the organizers have confirmed it in writing, pass --allow-control-templates.")
    params = ModelParams.from_json(best["model"])
    # Source weights come from the config (validation could not tune them: H1 was the held-out line). v1's config
    # gives H1 weight 0, so the submitted model is the validated K562-only one.
    weights_cfg = {str(k): float(w) for k, w in (cfg["model"].get("source_weights") or {}).items()}
    weights = {**params.source_weights, **weights_cfg}
    # The template: exactly the validated one. A parameter file written before template_sources existed has None
    # (= every source); fall back to its validation sources, which is what gamma was fitted for.
    if params.template_sources is None and best.get("validation_sources"):
        params.template_sources = [str(s) for s in best["validation_sources"]]
        print(f"note: {params_path.name} has no template_sources; using its validation sources "
              f"{params.template_sources} for the template")

    # ---- model inputs ----
    proc = P["processed_dir"]
    genes, panel, contexts = load_bundle_info(controls_dir)
    manifest = json.loads((controls_dir / "manifest.json").read_text())
    print(f"bundle {controls_dir} (panel {manifest.get('panel_id', '?')}): contexts {','.join(contexts)}, "
          f"{len(panel)} targets, {genes.size:,} genes")
    sources, reindexed = [], []
    for name in SUBMISSION_SOURCES:
        w = float(weights.get(name, 1.0))
        if w == 0:
            print(f"  {name}: weight 0 in the config's model.source_weights → not used")
            continue
        path = proc / f"{name}.npz"
        if not path.exists():
            sys.exit(f"missing {path} (source {name}, weight {w:g}): run scripts/01_build_sources.py")
        rs = ResponseSet.load(path)
        rs2 = to_bundle_axis(rs, genes, f"{controls_dir.name}/gene_names.csv")
        if rs2 is not rs:
            reindexed.append(name)
        sources.append(rs2)
        del rs
    if not sources:
        sys.exit("no source has a weight > 0 in the config's model.source_weights")
    params.source_weights = {s.name: float(weights.get(s.name, 1.0)) for s in sources}   # loaded sources only
    cis = CisModel.from_json((proc / "cis.json").read_text())
    tss = pd.read_csv(proc / "tss.csv")
    model = EffectModel(sources, params, cis=cis, tss=tss)          # params.use_cis switches cis on/off

    tsrc = [s.name for s in sources if params.template_sources is None or s.name in params.template_sources]
    beta_eff, gamma_eff = params.beta * params.beta_scale, params.gamma * params.gamma_scale
    print(f"settings from {params_path}: variant {best.get('variant')}, generator {gen_name}"
          f"{f' (spread {spread:g})' if gen_name == 'template' else ''}")
    print(f"  beta {params.beta:.3f} x {params.beta_scale:g} = {beta_eff:.3f}, gamma {params.gamma:.3f} x "
          f"{params.gamma_scale:g} = {gamma_eff:.3f}, min_called_lfc {params.min_called_lfc:g}, z0 {params.z0:g}, "
          f"z_soft {params.z_soft:g}, null_p {params.null_p:g}\n"
          f"  use_template {params.use_template}, use_specific {params.use_specific}, use_cis {params.use_cis}; "
          f"source weights {params.source_weights}; template from {tsrc or 'NO source'}")
    if params.use_template and not tsrc:
        print(f"WARNING: none of the template sources {params.template_sources} is loaded with a weight > 0, so "
              "the template is zero: this is not the validated model.")
    comps = model.components(panel)
    has = comps["has_source"]
    template = np.asarray(comps["template"], dtype=np.float64)
    del comps
    for rs in sources:
        print(f"  {rs.name}: measured {sum(t in set(rs.targets) for t in panel)}/{len(panel)} panel targets")
    print(f"  any source: {int(has.sum())}/{len(panel)}; the other {int((~has).sum())} get template + cis only")

    # ---- predictions and cells, context by context ----
    pred_stats = []
    gidx = {g: i for i, g in enumerate(genes)}
    # The template part of every prediction: gamma_eff * T(g), the same for all targets.
    tmpl_part = np.abs(gamma_eff * template) if (params.use_template and gamma_eff != 0) else np.zeros(genes.size)

    def predict_cpm(ctx, targets, ctrl_X):
        """Predicted mean CPM for the panel in context `ctx`, relative to ctx's own control mean CPM."""
        ctrl_cpm = mean_cpm(ctrl_X)
        lfc = model.predict_lfc(targets, ctrl_cpm)
        own = np.array([gidx.get(t, -1) for t in targets])
        off = np.ones(lfc.shape, dtype=bool)
        off[np.flatnonzero(own >= 0), own[own >= 0]] = False          # the knockdown itself is not a "change"
        changed = (np.abs(lfc) > 0.1) & off
        # Sanity check of the template: genes the scorer DE-tests in this context (control CPM > 5) that the
        # template alone moves by > 0.2 log2, i.e. for EVERY target. A K562-only template moves only a handful.
        n_tmpl = int(((tmpl_part > TEMPLATE_CHECK_LFC) & (ctrl_cpm > DE_GATE_CPM)
                      & (ctrl_cpm >= params.expr_floor_cpm)).sum())
        pred_stats.append({"context": ctx, "genes changed (|lfc|>0.1) per target": float(changed.sum(1).mean()),
                           "mean |lfc| on changed": float(np.abs(lfc[changed]).mean()) if changed.any() else 0.0,
                           f"template genes |γT|>{TEMPLATE_CHECK_LFC:g} (ctrl>{DE_GATE_CPM:g} CPM)": n_tmpl})
        print(f"  context {ctx}: the template alone moves {n_tmpl} DE-testable genes by > {TEMPLATE_CHECK_LFC:g} "
              "log2 for every target", flush=True)
        if n_tmpl > TEMPLATE_CHECK_WARN:
            print(f"  WARNING: that is more than {TEMPLATE_CHECK_WARN}: every target would get calls on those genes. "
                  "Check template_sources and gamma before submitting.", flush=True)
        return model.predict_cpm(targets, ctrl_cpm)

    def make_generator(ctrl_X):
        return TemplateGenerator(ctrl_X, spread) if gen_name == "template" else NBGenerator(ctrl_X)

    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\nbuilding {out_h5ad} ({cpp} cells per target) …", flush=True)
    build_submission(predict_cpm, make_generator, controls_dir, out_h5ad, out_h5ad.parent, cells_per_pert=cpp,
                     chunk_targets=int(S.get("chunk_targets", 50)), seed=int(S.get("seed", 0)))
    print(pd.DataFrame(pred_stats).set_index("context").to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"  ({time.time() - t0:,.0f} s)")

    # Provenance next to the file: what produced it (for the method description the rules require). This, not the
    # leaderboard name, is where the variant and the validation scores are recorded.
    prov_path = out_h5ad.with_suffix(".json")
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "controls_dir": str(controls_dir), "panel_id": manifest.get("panel_id"), "contexts": contexts,
        "cells_per_pert": cpp, "params_file": str(params_path), "variant": best.get("variant"),
        "model": json.loads(params.to_json()), "generator": gen_name, "spread": spread,
        "allow_control_templates": bool(args.allow_control_templates),
        "sources": [s.name for s in sources], "template_sources": tsrc, "config_source_weights": weights_cfg,
        "sources_reindexed_to_bundle_axis": reindexed, "validation_scores": best.get("cv_scores"),
        "template_check": pred_stats,
    }
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")

    # ---- checks and packaging ----
    print("\nvcc prep --dry-run …", flush=True)
    vcc_prep(out_h5ad, controls_dir, contexts, dry_run=True,
             cells_per_pert=cpp if cpp != CELLS_PER_PERT else None)
    print("vcc prep dry run: PASSED")
    if dry_run:
        print("dry run only: no .vcc package written" + (" (smoke)" if args.smoke else ""))
        return
    out_vcc = out_h5ad.with_suffix(".vcc")
    print(f"\npackaging {out_vcc} …", flush=True)
    vcc_prep(out_h5ad, controls_dir, contexts, out_vcc=out_vcc, dry_run=False)
    digest = sha256_file(out_vcc)
    sha_path = out_vcc.with_name(out_vcc.name + ".sha256")
    sha_path.write_text(f"{digest}  {out_vcc.name}\n")                 # `sha256sum -c` format
    prov.update({"vcc_file": out_vcc.name, "vcc_sha256": digest, "vcc_bytes": out_vcc.stat().st_size})
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"  {out_vcc.name}: {out_vcc.stat().st_size / 1e9:.2f} GB, sha256 {digest} (also in {sha_path.name})")
    if args.keep_h5ad:
        print(f"  kept {out_h5ad.name} (--keep-h5ad)")
    else:
        size = out_h5ad.stat().st_size
        out_h5ad.unlink()                                              # the .vcc holds everything that is submitted
        print(f"  deleted {out_h5ad.name} ({size / 1e9:.1f} GB; pass --keep-h5ad to keep it)")

    # A neutral leaderboard name: the leaderboard shows it to every team, so no method or score goes into it.
    name = f"{S.get('name', 'model-v1')}-{datetime.now(timezone.utc):%Y%m%d}"
    print("\nNOT submitted. After review, the team lead submits with:\n"
          f"  vcc submit {out_vcc} -m \"{name}\" --wait")


if __name__ == "__main__":
    main()
