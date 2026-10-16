# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: VCC (project .venv)
#     language: python
#     name: vcc
# ---

# %% [markdown]
# # VCC 2026 · Notebook 02 — A walkthrough of effect model v1
#
# **Why this notebook exists.** `scripts/01_build_sources.py` turns public screens into response tables, and
# `scripts/02_validate.py` picks model settings by running the official scorer on held-out H1 targets. Both are
# batch scripts: they print numbers, not understanding. This notebook opens the box. It loads what script 01
# produced, looks at the data the model learns from, fits the model's two gains by hand, and shows predicted vs
# measured fold changes on targets the fit never saw. Read it before arguing about the next model version.
#
# **The model in one line** (details: `docs/DESIGN_v1.md`; reasoning: `METHOD.md` §3). For target `t`, gene `g`,
# in a new cell line `C` of which we only know the control cells:
#
# ```
# predicted log2FC_C(t, g) = mask_C(g) · [ gamma · T(g)  +  beta · D(t, g) ]  +  cis(t, g)
# ```
#
# | piece | what it is | why it is there |
# |---|---|---|
# | `T(g)` **template** | the average response over the panel's targets, as measured in the source lines | every knockdown in a line shares part of its response (CRISPRi machinery, stress, proliferation); the scorer's 0 is exactly this average in the *new* line |
# | `D(t, g)` **specific response** | the source line's response minus its template, kept only where it is clearly above noise (`soft(z)`) | target-specific changes are what lift us above the 0 line; noisy small changes cost `jac` (notebook 01 §8) |
# | `gamma`, `beta` | two gains fitted by weighted least squares on a held-out line (H1) | responses shrink or grow when they move to another line; the fit learns by how much |
# | `mask_C(g)` | 0 for genes the new line barely expresses | no change can be measured (or scored) there |
# | `cis(t, g)` | a distance-to-TSS prior for genes next to the target | CRISPRi silences promoters near its target; used only for targets no source measured |
# | own gene | fixed `kd_lfc` (≈ 85% knockdown) | Arc kept only constructs with ≥ 80% knockdown; the scorer ignores this row anyway |
#
# **What each section shows**
#
# | § | Question | Output |
# |---|---|---|
# | 1–2 | Settings, loading the response tables | summary table |
# | 3 | How much of the 300-target panel does each source cover? | bar chart |
# | 4 | Are the z-scores calibrated? How sparse is a denoised response? | histograms |
# | 5 | How much of a response is template vs target-specific vs noise? Does the template transfer between lines? | bars, scatter |
# | 6 | How strong is the CRISPRi neighbour (cis) effect, by distance? | curve |
# | 7 | `fit_gains` on H1: how much do K562 responses shrink when they move to H1? | table |
# | 8 | Predicted vs measured log2FC on held-out H1 targets | scatters, summary |
# | 9 | How `ModelParams` map to the validation grid, and what script 02 chose | tables, charts |
#
# **Where to run it.** On TSUBAME, after `scripts/01_build_sources.py` (it needs `data/processed/*.npz`). It reads
# only the processed tables, so it is light: a few minutes and < 16 GB RAM (the K562 table is the largest piece;
# this notebook never expands it in full). §9 shows more once `scripts/02_validate.py` has written
# `work/validation/`. See `RUNBOOK.md` for running it unattended (`papermill`) and sending the result back.
#
# ⚠️ **Status.** v1 has been tested on fake data only. This notebook has never run on the real tables; its first
# run on the cluster is its test. If a cell fails, send the traceback (RUNBOOK.md, "What to send back").
#
# **This notebook is stored as a jupytext `.py`.** Make the paired `.ipynb` with
# `.venv/bin/jupytext --sync notebooks/02_effect_model_walkthrough.py` (commit both, as with notebook 01).

# %% [markdown]
# ## 1. Settings
#
# Paths can be overridden with environment variables (handy on TSUBAME, where the data may live in a shared
# group folder). `HOLDOUT` is the source that plays the "new cell line": it is **never** used as a training source
# below, only as the truth we compare against.

# %%
import dataclasses
import json
import os
import sys
from pathlib import Path


def find_project(start):
    """The project folder: the first parent (or the folder itself) that contains `vccpipe/`."""
    for p in [start, *start.parents]:
        if (p / "vccpipe").is_dir():
            return p
    raise FileNotFoundError("run this notebook from inside the project (no vccpipe/ found above the cwd)")


PROJECT = find_project(Path.cwd().resolve())
sys.path.insert(0, str(PROJECT))           # so `import vccpipe` works when the kernel starts in notebooks/

PROCESSED = Path(os.environ.get("VCC_PROCESSED_DIR", PROJECT / "data" / "processed"))   # output of script 01
CONTROLS_DIR = Path(os.environ.get("VCC_CONTROLS_DIR", PROJECT / "controls"))           # the 300-target panel
VALID_DIR = Path(os.environ.get("VCC_VALIDATION_DIR", PROJECT / "work" / "validation"))  # output of script 02
PROXY_TARGETS = Path(os.environ.get("VCC_PROXY_TARGETS", PROJECT / "work" / "h1_proxy" / "targets.txt"))  # script 01
CONFIG = Path(os.environ.get("VCC_CONFIG", PROJECT / "configs" / "v1.yaml"))            # validation grid
FIG_DIR = Path(os.environ.get("VCC_FIG_DIR", PROJECT / "work" / "figures" / "02_effect_model"))
HOLDOUT = os.environ.get("VCC_HOLDOUT", "h1_2025")      # the line we pretend we have never seen
PARAMS_FROM = os.environ.get("VCC_PARAMS_FROM", "config")   # settings for §4–8: "config" = the `model:` section
                                                        # of configs/v1.yaml; "best" = what script 02 chose
SEED = 0                                                # fixes the fit/evaluation split of H1 targets in §7–8
Z_SIG = 3.0                                             # "clearly changed" in the truth, for §8 (as notebook 01)

FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"PROJECT   = {PROJECT}\nPROCESSED = {PROCESSED}\nHOLDOUT   = {HOLDOUT}")
assert PROCESSED.is_dir() and any(PROCESSED.glob("*.npz")), (
    f"no response tables in {PROCESSED}: run scripts/01_build_sources.py first (RUNBOOK.md)")

# %% [markdown]
# ## 2. Imports, plotting style, and the response tables
#
# Each `data/processed/<name>.npz` is one **`ResponseSet`**: for every target a source measured, the mean
# expression (CPM) of its knockdown cells on the VCC 18,533-gene axis, the source's control mean, and the
# standard error of each log2 fold change. `lfc()` and `z()` derive fold changes and z-scores from those.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vccpipe.cis import CisModel
from vccpipe.common import DE_GATE_CPM, load_bundle_info
from vccpipe.model import EffectModel, ModelParams, fit_gains
from vccpipe.sources import ResponseSet

try:                                     # `display` exists inside Jupyter; fall back to print elsewhere
    display
except NameError:
    display = print

# One quiet style for every chart: hairline axes, light grid, one accent colour (blue) for the data, grey for
# context and reference lines, dark ink for text. No rainbow colour maps anywhere.
BLUE, GREY, INK, INK2 = "#2a78d6", "#898781", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False,
})


def finish(fig, name):
    """Tidy, save a PNG copy (so the figures can be sent back without the notebook), and show."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.show()


# The VCC gene axis and the validation panel (300 targets) come from the controls bundle.
genes, panel, contexts = load_bundle_info(CONTROLS_DIR)
gidx = {g: i for i, g in enumerate(genes)}

sources = {}
for path in sorted(PROCESSED.glob("*.npz")):
    rs = ResponseSet.load(path)
    assert list(rs.genes) == list(genes), f"{path.name} is not on the VCC gene axis"
    sources[rs.name] = rs
assert HOLDOUT in sources, f"holdout '{HOLDOUT}' not among the loaded sources {list(sources)}"
truth = sources[HOLDOUT]
train = [rs for name, rs in sources.items() if name != HOLDOUT]      # never train on the line we evaluate on
train_names = [rs.name for rs in train]
print(f"training sources: {train_names}; held-out line: {HOLDOUT}")


def summarize(rs):
    """One row of facts per source: size, depth of measurement, knockdown, how many targets respond."""
    m = rs.meta
    p = m["phenotype_p"]
    return {
        "targets": len(rs.targets),
        "genes measured": int(rs.measured.sum()),
        "genes expressed (ctrl CPM > 5)": int((rs.ctrl_cpm > DE_GATE_CPM).sum()),
        "median cells / target": float(m["n_cells"].median()),
        "median knockdown left": float(m["kd_remaining"].median()),
        "targets with phenotype p < 0.05": int((p < 0.05).sum()) if p.notna().any() else np.nan,
    }


display(pd.DataFrame({name: summarize(rs) for name, rs in sources.items()}).T)

# Model settings. The scripts start from the `model:` section of configs/v1.yaml; we do the same, so the numbers
# below match what scripts 02/03 would do before tuning. (PyYAML ships with the project environment: jupytext and
# papermill depend on it. Without it we fall back to the ModelParams defaults, which match the config.)
try:
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text()) if CONFIG.exists() else {}
except ImportError:
    cfg = {}


def best_model_settings(d):
    """The ModelParams part of best_params.json: script 02 stores it under "model" (with the gains refitted on
    all proxy targets); a flat file is accepted too."""
    m = d.get("model", d)
    return {f.name: m[f.name] for f in dataclasses.fields(ModelParams) if f.name in m}


P_CONFIG = ModelParams.from_json(cfg["model"]) if cfg.get("model") else ModelParams()
P0 = P_CONFIG
best_json = VALID_DIR / "best_params.json"
if PARAMS_FROM == "best" and best_json.exists():
    P0 = dataclasses.replace(P_CONFIG, **best_model_settings(json.loads(best_json.read_text())))
print(f"model settings for §4–8 ({PARAMS_FROM}):", dataclasses.asdict(P0))

# %% [markdown]
# **How to read it.** *Median knockdown left* should be ≈ 0.1–0.3 (CRISPRi typically leaves 10–30% of the
# target's expression). *Phenotype p* is Replogle's energy-distance test ("does this knockdown change the
# transcriptome at all?"); it is unknown (nan) for H1. Targets with `phenotype_p > null_p` get **no specific
# response** in the model, so that count is roughly how many targets the model predicts as more than template.

# %% [markdown]
# ## 3. Coverage of the 300-target panel
#
# The model has a target-specific response only for panel targets that some source measured. For the others it
# predicts template + cis prior + knockdown, i.e. "the line's average response". Coverage therefore caps how
# far above 0 we can get. (The final phase brings a **new** panel: run the notebook with
# `VCC_CONTROLS_DIR=controls_DEF` then.)

# %%
cover = pd.DataFrame({name: [t in set(rs.targets) for t in panel] for name, rs in sources.items()}, index=panel)
cover["any training source"] = cover[train_names].any(axis=1) if train_names else False
counts = cover.sum().astype(int)
counts["none"] = int((~cover[list(sources)].any(axis=1)).sum())

fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(counts) + 1))
y = np.arange(len(counts))[::-1]
ax.barh(y, counts.values, color=BLUE, height=0.55)
ax.set_yticks(y, counts.index)
for yi, v in zip(y, counts.values):
    ax.text(v + 3, yi, str(v), va="center", fontsize=8, color=INK)
ax.set_xlim(0, len(panel) * 1.1)
ax.set_xlabel(f"panel targets covered (of {len(panel)})")
ax.set_title("Which panel targets does each source measure?")
ax.grid(axis="y", visible=False)
finish(fig, "03_panel_coverage")

# Covered is not the same as useful: a target can be measured in few cells or show no detectable response.
for rs in train:
    m = rs.meta.reindex([t for t in panel if t in set(rs.targets)])
    detectable = (m["phenotype_p"] < 0.05).sum() if m["phenotype_p"].notna().any() else "unknown"
    print(f"{rs.name}: {len(m)} panel targets measured; median {m['n_cells'].median():.0f} cells; "
          f"detectable response (phenotype p < 0.05): {detectable}")

# %% [markdown]
# ## 4. Are the z-scores calibrated? How sparse is a denoised response?
#
# The specific response keeps a gene with weight `soft(z) = clip((|z| − z0) / z_soft, 0, 1)`: nothing below
# `|z| = z0`, full weight above `z0 + z_soft`. That only works if `z` means what it says. For genes a knockdown
# does *not* affect, `z` should follow a standard normal: if the bulk of the histogram is much wider than the
# grey reference curve, the standard errors are too small and noise leaks into the prediction (raise `z0`);
# if it is much narrower, real effects get thrown away.
#
# We look at the first training source (usually K562 genome-wide), on panel targets it measured, genes it
# expresses (control CPM > 5), and with each target's own gene removed.

# %%
src = train[0]
sel = [t for t in panel if t in set(src.targets)]
if len(sel) < 50:                                       # tiny overlap → use a random sample of the source instead
    sel = [str(t) for t in np.random.default_rng(SEED).choice(src.targets, min(300, len(src.targets)),
                                                                replace=False)]
sub = src.subset(sel)
z = np.asarray(sub.z(), dtype=np.float64)
ok = np.broadcast_to(src.measured & (src.ctrl_cpm > DE_GATE_CPM), z.shape).copy()
own = np.array([gidx.get(t, -1) for t in sub.targets])
ok[np.flatnonzero(own >= 0), own[own >= 0]] = False     # the knockdown itself is not a "response"
zv = z[ok]
zv = zv[np.isfinite(zv)]

fig, ax = plt.subplots(figsize=(6.5, 3.4))
edges = np.linspace(-10, 10, 161)
ax.hist(np.clip(zv, -10, 10), bins=edges, color=BLUE, label=f"{src.name}: measured z")
centers = 0.5 * (edges[1:] + edges[:-1])
expected = zv.size * np.diff(edges) * np.exp(-centers ** 2 / 2) / np.sqrt(2 * np.pi)
ax.plot(centers, expected, color=GREY, lw=1.5, label="if every gene were unaffected: N(0, 1)")
for s in (-1, 1):
    ax.axvline(s * P0.z0, color=INK2, lw=0.8)
ax.text(P0.z0 + 0.3, 0.96, f"±z0 = {P0.z0:g}", transform=ax.get_xaxis_transform(), fontsize=8, color=INK2,
        va="top")
ax.set_yscale("log")
ax.set_ylim(bottom=1)
ax.set_xlabel("z = log2FC / standard error (clipped at ±10)")
ax.set_ylabel("gene × target pairs (log scale)")
ax.set_title(f"Calibration of z on {len(sel)} targets of {src.name}")
ax.legend(loc="upper left", fontsize=8)
finish(fig, "04_z_calibration")

within1 = np.mean(np.abs(zv) < 1)
print(f"share of |z| < 1: {within1:.2f} (0.68 if all genes were unaffected and z were exact)")
print(f"share of |z| > z0={P0.z0}: {np.mean(np.abs(zv) > P0.z0):.3f} (0.046 expected from noise alone)")

# %% [markdown]
# Per target: how many genes would the denoised response actually change? This is roughly how many genes the
# generator will move for that target, so it drives `jac` (too many = over-calling) and `fid` (too few =
# under-calling). Full weight starts at `|z| ≥ z0 + z_soft`.

# %%
n_kept = (np.where(ok, np.abs(z), 0) >= P0.z0 + P0.z_soft).sum(axis=1)
fig, ax = plt.subplots(figsize=(6.5, 3.0))
ax.hist(n_kept, bins=np.unique(np.geomspace(1, max(n_kept.max(), 2) + 1, 40).astype(int)) - 1, color=BLUE)
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel(f"genes per target with |z| ≥ z0 + z_soft = {P0.z0 + P0.z_soft:g}")
ax.set_ylabel("targets")
ax.set_title(f"How many genes the specific response keeps ({src.name}, panel targets)")
finish(fig, "04_genes_kept_per_target")
print(f"median {np.median(n_kept):.0f} genes per target; "
      f"{np.mean(n_kept < 10):.0%} of targets keep fewer than 10 (nearly null)")

# %% [markdown]
# ## 5. Template vs target-specific response vs noise
#
# For one source and a set of targets, every measured fold change splits exactly into
# `lfc(t, g) = T(g) + [lfc(t, g) − T(g)]`, with `T(g)` the mean over the targets. Summed over targets and expressed
# genes (control CPM > 5, each target's own gene excluded), the squares split exactly into a **template** share and
# a **specific** share (they add up to 1). Part of the specific share is just measurement **noise**: the summed
# squared standard errors tell us how much to expect from noise alone, drawn as a grey tick on the specific bar.
# A specific bar that barely clears its tick means the target-specific responses are mostly noise at this depth
# (then `z0` matters a lot). If the tick sits *above* the bar, the standard errors overstate the noise.
#
# For reference, Molina & Zhang (2026) estimate across lines: template ~28%, conserved per-target ~29%,
# line × target interaction ~24%, noise ~19%. The table's last column is the template's share of the
# noise-corrected signal (nan when the expected noise is ≥ 75% of the total: too noisy to split).

# %%
def variance_split(rs, targets):
    """Template / specific shares of the summed squared log2FC, the expected noise share, and T(g).

    Shares are of the raw sum over (target, gene) pairs with an expressed, measured gene other than the target.
    Noise = summed squared standard errors over the same pairs; 1/T of it ends up in the template (a mean of T
    noisy values), the rest in the specific part.
    """
    sub = rs.subset(targets)
    lfc = np.asarray(sub.lfc(), dtype=np.float64)
    se = np.asarray(sub.se_lfc, dtype=np.float64)
    ok = np.broadcast_to(rs.measured & (rs.ctrl_cpm > DE_GATE_CPM), lfc.shape).copy()
    own = np.array([gidx.get(t, -1) for t in sub.targets])
    ok[np.flatnonzero(own >= 0), own[own >= 0]] = False
    ok &= np.isfinite(se) & np.isfinite(lfc)
    n = np.maximum(ok.sum(axis=0), 1)
    tmpl = np.where(ok, lfc, 0).sum(axis=0) / n                     # T(g) over the targets
    template = np.sum(np.where(ok, tmpl, 0) ** 2)
    specific = np.sum(np.where(ok, lfc - tmpl, 0) ** 2)
    total = template + specific
    noise = np.sum(np.where(ok, se, 0) ** 2)
    n_t = len(sub.targets)
    signal = total - noise
    return pd.Series({"template": template / total, "specific": specific / total,
                      "expected noise": noise / total,
                      # unstable when little signal is left after removing noise: report only above 25%
                      "template share of signal": ((template - noise / n_t) / signal if signal > 0.25 * total
                                                   else np.nan)}), tmpl


splits, templates = {}, {}
for name, rs in sources.items():
    tg = [t for t in panel if t in set(rs.targets)]
    if len(tg) < 20:                                    # few panel targets (e.g. H1): use its own targets
        tg = list(rs.targets)
    splits[name], templates[name] = variance_split(rs, tg)
    print(f"{name}: {len(tg)} targets")
splits = pd.DataFrame(splits).T
display(splits.round(3))

fig, axes = plt.subplots(1, len(splits), figsize=(3.2 * len(splits), 2.9), sharey=True, squeeze=False)
for ax, (name, row) in zip(axes[0], splits.iterrows()):
    vals = row[["template", "specific"]]
    ax.bar([0, 1], vals.values, color=BLUE, width=0.5)
    for x, v in enumerate(vals.values):
        ax.text(x, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.hlines(row["expected noise"], 0.68, 1.32, color=GREY, lw=2)       # noise level on the specific bar
    ax.text(1.36, row["expected noise"], "noise", va="center", fontsize=7, color=INK2)
    ax.set_xticks([0, 1], ["template", "specific"])
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(0, max(1.05, row["expected noise"] + 0.1))
    ax.set_title(name)
    ax.grid(axis="x", visible=False)
axes[0, 0].set_ylabel("share of summed squared log2FC")
finish(fig, "05_variance_split")

# %% [markdown]
# **Does the template transfer between lines?** When the model predicts H1, its template is K562's average
# response over *H1's* targets (the panel being predicted). The scorer's 0 line is H1's own average response.
# The closer these two agree, the closer `gamma · T` gets us to 0 for free; the rest must come from the
# specific part. (v1 does not yet predict the template from the new line's controls; that is METHOD §3a.)

# %%
# The H1 "panel" is the proxy's target list (script 02 predicts and scores exactly these); before script 01 has built
# the proxy, every H1 target is used.
if PROXY_TARGETS.exists():
    h1_targets = [t for t in PROXY_TARGETS.read_text().split() if t in set(truth.targets)]
    print(f"H1 panel: {len(h1_targets)} proxy targets from {PROXY_TARGETS.name}")
else:
    h1_targets = list(truth.targets)
    print(f"H1 panel: all {len(h1_targets)} H1 targets ({PROXY_TARGETS} not found)")
both = truth.measured & (truth.ctrl_cpm > DE_GATE_CPM)
for rs in train:
    shared = [t for t in h1_targets if t in set(rs.targets)]
    if len(shared) < 10:
        print(f"{rs.name}: only {len(shared)} H1 targets measured; skipped")
        continue
    _, t_src = variance_split(rs, shared)
    _, t_true = variance_split(truth, shared)
    g_ok = both & rs.measured & (rs.ctrl_cpm > DE_GATE_CPM)
    r = np.corrcoef(t_src[g_ok], t_true[g_ok])[0, 1]
    slope = np.polyfit(t_src[g_ok], t_true[g_ok], 1)[0]
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(t_src[g_ok], t_true[g_ok], s=4, color=BLUE, alpha=0.35, linewidths=0)
    lim = max(np.nanpercentile(np.abs(np.r_[t_src[g_ok], t_true[g_ok]]), 99.9), 0.1)
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"template from {rs.name} (mean log2FC over {len(shared)} shared targets)")
    ax.set_ylabel(f"true template in {HOLDOUT}")
    ax.set_title(f"Template transfer: r = {r:.2f}, slope = {slope:.2f}")
    finish(fig, f"05_template_transfer_{rs.name}")

# %% [markdown]
# ## 6. The CRISPRi neighbour (cis) effect by distance
#
# dCas9-KRAB silences chromatin around its target, so genes whose TSS lies within a few kb of the target's TSS
# often go down too. The scorer excludes only the target's own row, so these neighbours count. Script 01 fitted
# the median log2FC per distance bin (`data/processed/cis.json`); below it is refitted on each source as a check
# that the curve is a property of CRISPRi, not of one cell line. Expect a clear negative value in the first bins
# (< 1–2 kb, often bidirectional promoters) fading to ~0 by 10–20 kb.

# %%
tss = pd.read_csv(PROCESSED / "tss.csv")
cis = CisModel.from_json((PROCESSED / "cis.json").read_text())
curves = {"cis.json (used by the model)": cis}
for name, rs in sources.items():
    curves[f"refit on {name}"] = CisModel().fit(rs, tss)

display(pd.concat({label: cm.summary().set_index(["lo_kb", "hi_kb"]) for label, cm in curves.items()}, axis=1)
        .round(3))

fig, axes = plt.subplots(1, len(curves), figsize=(3.4 * len(curves), 3.0), sharey=True, squeeze=False)
for ax, (label, cm) in zip(axes[0], curves.items()):
    b = list(cm.bins_kb)
    x = np.arange(len(b) - 1)
    ax.plot(x, np.array(cm.effect, dtype=float), color=BLUE, marker="o", ms=5)   # None (empty bin) → gap
    ax.axhline(0, color=INK2, lw=0.8)
    ticks = [f"{b[i]:g}–{b[i + 1]:g}\nn={cm.n_pairs[i]}" for i in x]
    ax.set_xticks(x, ticks, fontsize=7)
    ax.set_title(label)
    ax.set_xlabel("TSS distance (kb), pairs")
axes[0, 0].set_ylabel("median log2FC of the neighbour gene")
finish(fig, "06_cis_curve")

cis_panel = np.asarray(cis.predict(panel, genes, tss))
n_cis = (cis_panel != 0).sum(axis=1)
print(f"panel targets with a cis prior on ≥ 1 gene: {(n_cis > 0).sum()} of {len(panel)}; "
      f"median {np.median(n_cis[n_cis > 0]) if (n_cis > 0).any() else 0:.0f} genes each. "
      "The model applies it only to targets no source measured.")

# %% [markdown]
# ## 7. `fit_gains` on H1: how much do responses shrink between lines?
#
# `beta` (specific) and `gamma` (template) are the only fitted numbers in v1. `fit_gains` regresses H1's measured
# log2FC on the model's two components for H1 targets, weighting each (target, gene) pair by `1 / se²` (capped),
# over genes H1 expresses, excluding the target's own gene.
#
# We split the H1 targets that a training source covers into two halves: gains are fitted on the **fit** half and
# judged in §8 on the **evaluation** half. Script 02 does the same with 2-fold cross-fitting over all proxy targets
# and the official scorer, then refits the gains on all proxy targets for the submission. Fitting each half
# separately also shows how stable the gains are.
#
# * `gamma ≈ 1`: the source template transfers at full size. `gamma ≈ 0`: it does not transfer.
# * `beta ≈ 1`: specific responses keep their size. `beta < 1`: they shrink in the new line (or are partly noise).
#   Negative values are a red flag (the source points the wrong way on average).

# %%
covered = [t for t in h1_targets if any(t in set(rs.targets) for rs in train)]
assert len(covered) >= 10, f"only {len(covered)} H1 targets are covered by the training sources"
perm = np.random.default_rng(SEED).permutation(len(covered))
fit_t = [covered[i] for i in sorted(perm[: len(covered) // 2])]
eval_t = [covered[i] for i in sorted(perm[len(covered) // 2:])]

model0 = EffectModel(train, P0, cis=cis, tss=tss)
gains = {}
for label, tg in [("all covered", covered), ("fit half", fit_t), ("evaluation half", eval_t)]:
    # `targets=` is the panel the template is averaged over: always the full H1 panel, exactly as at prediction
    # time. Which targets enter the regression is set by the truth we pass: H1 restricted to this half.
    b, g = fit_gains(model0, truth.subset(tg), targets=h1_targets)
    gains[label] = {"targets": len(tg), "beta (specific)": b, "gamma (template)": g}
display(pd.DataFrame(gains).T.round(3))

beta, gamma = gains["fit half"]["beta (specific)"], gains["fit half"]["gamma (template)"]
P_fit = dataclasses.replace(P0, beta=float(beta), gamma=float(gamma))
print(f"§8 uses the fit-half gains: beta = {beta:.3f}, gamma = {gamma:.3f}")

# %% [markdown]
# ## 8. Predicted vs measured log2FC on held-out H1 targets
#
# We predict **every** H1 target in one call (the template is an average over the panel being predicted, so
# predicting a handful of targets alone would give a different template), then look only at the evaluation half.
# Truth = H1's measured log2FC; "clearly changed" = `|z| > 3` in H1 (the genes the scorer's Wilcoxon test is most
# likely to call). We exclude each target's own gene and genes H1 does not express (control CPM ≤ 5, which the
# scorer does not test).

# %%
def evaluate(params, label):
    """Per-target agreement between the model's prediction and H1's measured response (evaluation half)."""
    m = EffectModel(train, params, cis=cis, tss=tss)
    pred = np.asarray(m.predict_lfc(h1_targets, truth.ctrl_cpm), dtype=np.float64)
    row = {t: i for i, t in enumerate(h1_targets)}
    sub = truth.subset(eval_t)
    true, zt = np.asarray(sub.lfc(), dtype=np.float64), np.asarray(sub.z(), dtype=np.float64)
    expressed = truth.measured & (truth.ctrl_cpm > DE_GATE_CPM)
    out = []
    for i, t in enumerate(sub.targets):
        g_ok = expressed.copy()
        if t in gidx:
            g_ok[gidx[t]] = False
        p, y = pred[row[t]][g_ok], true[i][g_ok]
        sig = np.abs(zt[i][g_ok]) > Z_SIG
        out.append({"target": t, "model": label, "true changed genes": int(sig.sum()),
                    "sign agreement": np.mean(np.sign(p[sig]) == np.sign(y[sig])) if sig.any() else np.nan,
                    "pearson r": np.corrcoef(p, y)[0, 1] if p.std() > 0 and y.std() > 0 else np.nan})
    return pd.DataFrame(out), pred


per_target, pred_full = evaluate(P_fit, "template + specific")
per_target = per_target.set_index("target")
print(per_target.describe().round(3))

# %% [markdown]
# Four held-out targets with the strongest measured responses. Grey = all expressed genes, blue = genes clearly
# changed in H1. Points on the diagonal are perfect; points on the horizontal zero line are genes the model
# predicts as unchanged (the denoising at work); the wrong-sign quadrants (top-left, bottom-right) cost `fid`.

# %%
show = per_target.sort_values("true changed genes", ascending=False).index[:4].tolist()
row = {t: i for i, t in enumerate(h1_targets)}
sub = truth.subset(show)
true_s, z_s = np.asarray(sub.lfc(), dtype=np.float64), np.asarray(sub.z(), dtype=np.float64)
expressed = truth.measured & (truth.ctrl_cpm > DE_GATE_CPM)
fig, axes = plt.subplots(2, 2, figsize=(8, 7.4), squeeze=False)
for ax, (i, t) in zip(axes.ravel(), enumerate(sub.targets)):
    g_ok = expressed.copy()
    if t in gidx:
        g_ok[gidx[t]] = False
    x, y, sig = true_s[i][g_ok], pred_full[row[t]][g_ok], np.abs(z_s[i][g_ok]) > Z_SIG
    ax.scatter(x[~sig], y[~sig], s=3, color=GREY, alpha=0.3, linewidths=0, label="other expressed genes")
    ax.scatter(x[sig], y[sig], s=8, color=BLUE, alpha=0.8, linewidths=0, label=f"|z| > {Z_SIG:g} in H1")
    lim = max(np.nanpercentile(np.abs(x), 99.9), np.nanpercentile(np.abs(y), 99.9), 0.5)
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8)
    ax.axhline(0, color="#c3c2b7", lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    st = per_target.loc[t]
    ax.set_title(f"{t}: r = {st['pearson r']:.2f}, sign agreement = {st['sign agreement']:.0%}")
    ax.set_xlabel("measured log2FC in H1")
    ax.set_ylabel("predicted log2FC")
axes[0, 0].legend(loc="upper left", fontsize=7, markerscale=2)
finish(fig, "08_pred_vs_true_examples")

# %% [markdown]
# **Across all held-out targets, which part of the model helps?** We switch components off one at a time and
# measure, per target, the share of clearly changed genes whose predicted sign is right (a predicted 0 counts as
# wrong, as in the scorer's `fid`). The last bar is **H1's own true average response** given to every target:
# that is the scorer's baseline, i.e. the 0 of the scale. It uses the answers (including each target's own
# response, as the scorer's baseline does), so it is a reference line, not a model. To score above 0 on
# direction-based metrics, the model has to beat it.

# %%
variants = {
    "template only": dataclasses.replace(P_fit, use_specific=False),
    "specific only": dataclasses.replace(P_fit, use_template=False),
    "template + specific": P_fit,
}
summary = {}
for label, params in variants.items():
    df = per_target if label == "template + specific" else evaluate(params, label)[0]   # reuse the full run
    summary[label] = df["sign agreement"].median()

# Reference: H1's true mean response over its own panel (the scorer's 0), scored the same way.
h1_all = truth.subset(h1_targets)
oracle = np.asarray(h1_all.lfc(), dtype=np.float64).mean(axis=0)
sub = truth.subset(eval_t)
true_e, z_e = np.asarray(sub.lfc(), dtype=np.float64), np.asarray(sub.z(), dtype=np.float64)
agree = []
for i, t in enumerate(sub.targets):
    g_ok = expressed.copy()
    if t in gidx:
        g_ok[gidx[t]] = False
    sig = np.abs(z_e[i][g_ok]) > Z_SIG
    if sig.any():
        agree.append(np.mean(np.sign(oracle[g_ok][sig]) == np.sign(true_e[i][g_ok][sig])))
summary[f"{HOLDOUT} true mean (scorer's 0)"] = np.median(agree) if agree else np.nan
summary = pd.Series(summary)
display(summary.round(3).to_frame("median sign agreement"))

fig, ax = plt.subplots(figsize=(6.5, 2.6))
yy = np.arange(len(summary))[::-1]
ax.barh(yy, summary.values, color=BLUE, height=0.55)
ax.set_yticks(yy, summary.index)
for yi, v in zip(yy, summary.values):
    ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
ax.set_xlim(0, 1.08)
ax.axvline(0.5, color=GREY, lw=0.8)
ax.set_xlabel(f"median share of clearly changed genes with the right sign (held-out {HOLDOUT} targets)")
ax.grid(axis="y", visible=False)
finish(fig, "08_component_ablation")

fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.hist(per_target["pearson r"].dropna(), bins=30, color=BLUE)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("per-target Pearson r, predicted vs measured log2FC (expressed genes)")
ax.set_ylabel("held-out targets")
ax.set_title("Template + specific, fit-half gains")
finish(fig, "08_pearson_r_hist")

# %% [markdown]
# ## 9. `ModelParams` and the validation grid
#
# `ModelParams` holds every setting of the effect model. Two are *fitted* (`beta`, `gamma`, §7); the others are
# *chosen* by `scripts/02_validate.py`, which builds predictions for a grid of settings, turns them into cells
# with the generator, and scores each with the official scorer on held-out H1 targets
# (`work/validation/results.csv`, best row → `best_params.json`). The table says which metric each setting is
# expected to move, so a surprising winner can be sanity-checked.

# %%
EXPLAIN = {
    "z0": ("specific response: |z| below this is treated as noise", "higher → fewer genes changed: jac ↑ (less over-calling), fid/nmae ↓ if real effects are dropped"),
    "z_soft": ("width of the ramp from 0 to full weight above z0", "larger → gentler shrinkage of borderline genes"),
    "beta": ("gain on the specific response (fitted on H1)", "pds, mse, nmae; too small → too few significant genes (fid)"),
    "gamma": ("gain on the template (fitted on H1)", "mse and every DE metric on the template genes"),
    "null_p": ("targets with phenotype p above this get no specific response", "lower → more targets predicted as template only: jac ↑ on truly null targets"),
    "expr_floor_cpm": ("genes below this control CPM in the new line are left unchanged", "mostly safety; the scorer tests only genes > 5 CPM"),
    "kd_lfc": ("log2FC of the target's own gene", "not scored (own row excluded); keeps cells realistic"),
    "use_template": ("switch the template on/off", "ablation"),
    "use_specific": ("switch the specific response on/off", "ablation"),
    "use_cis": ("switch the cis prior on/off (targets without a source only)", "ablation"),
    "source_weights": ("relative weight of each source (missing → 1)", "matters once there are several sources"),
}
defaults = dataclasses.asdict(P_CONFIG)
display(pd.DataFrame({"configs/v1.yaml": {k: str(v) for k, v in defaults.items()},
                      "what it does": {k: EXPLAIN.get(k, ("", ""))[0] for k in defaults},
                      "expected to move": {k: EXPLAIN.get(k, ("", ""))[1] for k in defaults}}))

# The grid itself lives in configs/v1.yaml (shown here only; the scripts are the authority). Besides ModelParams
# fields it varies the cell generator ("template" edits real control cells, "nb" samples them) and its `spread`.
if cfg.get("validation"):
    print(f"validation section of {CONFIG.name}:")
    print(json.dumps(cfg["validation"], indent=2, default=str))

# %% [markdown]
# After `scripts/02_validate.py` has run, the scores per grid point. **Scaled scores**: 0 = H1's true average
# response given to every target (the oracle baseline), 1 = a split-half replicate of the real data. Each chart
# below shows, for one setting, the best overall score reached at each of its values (other settings free). A
# winner at the edge of the grid means the grid should be extended; a flat line means the setting barely matters.

# %%
results_csv = VALID_DIR / "results.csv"
if results_csv.exists():
    res = pd.read_csv(results_csv)
    cols = [c for c in ["variant", "kind", "overall", "pds", "mse", "nmae", "fid", "reach", "jac",
                        "beta_mean", "gamma_mean"] if c in res.columns]
    ranked = res.sort_values("overall", ascending=False)
    display(ranked[cols].round(3))

    # Every variant's overall score, best first; the vertical line is the "predict no change" reference.
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(ranked) + 1.2))
    yy = np.arange(len(ranked))[::-1]
    is_ref = (ranked["kind"] == "reference").to_numpy() if "kind" in ranked.columns else np.zeros(len(ranked), bool)
    ax.barh(yy, ranked["overall"], color=np.where(is_ref, GREY, BLUE), height=0.6)   # grey = the reference
    ax.set_yticks(yy, ranked["variant"], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ref = res.loc[res.get("kind", pd.Series(dtype=str)) == "reference", "overall"]
    if len(ref):
        ax.axvline(ref.iloc[-1], color=GREY, lw=1.5)
        ax.text(ref.iloc[-1], 1.0, " control_paste", transform=ax.get_xaxis_transform(), fontsize=7,
                color=INK2, va="bottom")
    ax.set_xlabel("overall score on the H1 proxy (scaled: 0 = context-mean baseline, 1 = replicate)")
    ax.grid(axis="y", visible=False)
    finish(fig, "09_variants_overall")

    # Marginals over the grid rows only (ablations and the reference are single points, shown above).
    grid_rows = res[res["kind"] == "grid"] if "kind" in res.columns else res
    knobs = list(defaults) + ["generator", "spread"]
    varied = [c for c in knobs if c in grid_rows.columns and grid_rows[c].astype(str).nunique() > 1]
    print(f"{len(grid_rows)} grid variants; settings varied: {varied}")
    if varied:
        fig, axes = plt.subplots(1, len(varied), figsize=(2.6 * len(varied), 2.8), sharey=True, squeeze=False)
        for ax, c in zip(axes[0], varied):
            best = grid_rows.groupby(grid_rows[c].astype(str))["overall"].max()
            try:                                        # numeric settings in numeric order
                best = best.iloc[np.argsort(best.index.astype(float))]
            except ValueError:
                pass
            ax.plot(np.arange(len(best)), best.values, color=BLUE, marker="o", ms=5)
            ax.set_xticks(np.arange(len(best)), best.index, fontsize=7)
            ax.set_xlim(-0.5, len(best) - 0.5)
            ax.set_xlabel(c)
            ax.axhline(0, color=INK2, lw=0.8)
        axes[0, 0].set_ylabel("best overall score (scaled)")
        finish(fig, "09_grid_marginals")
else:
    print(f"{results_csv} not found: run scripts/02_validate.py, then rerun this section.")

if best_json.exists():
    best = json.loads(best_json.read_text())
    P_best = dataclasses.replace(P_CONFIG, **best_model_settings(best))
    print(f"\nChosen by validation: variant {best.get('variant')}, generator {best.get('generator')}, "
          f"spread {best.get('spread')}")
    for k, v in dataclasses.asdict(P_best).items():
        if v != defaults[k]:
            print(f"  {k}: {defaults[k]} (configs/v1.yaml) → {v}")
    if best.get("cv_gains_per_fold"):
        print(f"  gains (beta, gamma) per cross-fitting fold: {best['cv_gains_per_fold']}")
    for label, key in [("chosen variant", "cv_scores"), ("control_paste", "control_paste_scores")]:
        if best.get(key):
            print(f"  {label}: " + ", ".join(f"{m} {v:.3f}" for m, v in best[key].items()))

# %% [markdown]
# ## 10. What to look for (and report back)
#
# * **§4**: the z histogram far wider than N(0,1) near 0 → standard errors too small; expect `z0` to be pushed up
#   by validation. Most targets keeping < 10 genes → the panel is mostly near-null in the source; `jac` will
#   reward calling little.
# * **§5**: a small template share in K562 but a large one in H1 (or the reverse) means the generic response
#   is line-specific; a template-transfer `r` well below ~0.5 says `gamma · T` cannot reach the baseline, and
#   predicting the template from control cells (METHOD §3a) becomes the next priority.
# * **§6**: a cis curve that is not clearly negative in the first bin → check TSS matching (gene symbols vs
#   GENCODE names) before trusting it.
# * **§7**: gains that differ a lot between the two halves → the fit is driven by a few strong targets; prefer the
#   all-covered fit for the submission and say so in the PR.
# * **§8**: if "template + specific" does not beat the scorer's-0 bar on sign agreement, the specific part adds
#   noise rather than signal on H1; compare with `work/validation/results.csv` before submitting anything.
#
# Figures are saved to `work/figures/02_effect_model/` for sharing.
