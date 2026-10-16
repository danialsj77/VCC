"""Tests of the model-v1 pipeline: the H1 proxy builder, the config, the scripts' helpers (proxy strength filter,
validation stages and --resume, the submission's safeguards), the TSUBAME batch script's logic, and the whole
01 → 02 → 03 chain on fake data.

Run either way (from the project folder):
    .venv/bin/python tests/test_end_to_end.py                      # every test, PASS/FAIL per test
    .venv/bin/python tests/test_end_to_end.py test_proxy_format_and_filters   # only the named test(s)
    pytest tests/test_end_to_end.py                                # if pytest is installed

Nothing is downloaded and nothing under data/ is read. Fake datasets with the real files' layout (tests/fakes.py)
are written into a temporary folder; the end-to-end test uses the REAL VCC gene axis and controls bundle
(controls/) so the submission is checked against the real format by `vcc prep --dry-run`.

`test_pipeline_end_to_end` takes a few minutes (CPU scorer) and ~4-8 GB of RAM; the other tests take seconds.
"""

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fakes                                           # noqa: E402
from vccpipe.common import CONTROL                     # noqa: E402
from vccpipe.proxy import build_h1_proxy               # noqa: E402

PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)
CONTROLS = ROOT / "controls"
CONFIG = ROOT / "configs" / "v1.yaml"


class Skipped(Exception):
    """Raised when a test cannot run here (e.g. no controls bundle); pytest sees a skip instead."""


def _skip(msg):
    if "pytest" in sys.modules:                        # running under pytest: report a proper skip
        import pytest
        pytest.skip(msg)
    raise Skipped(msg)                                 # running directly: the __main__ runner prints SKIP


def _load_script(name):
    """Import a script from scripts/ as a module (their names start with digits, so `import` can't)."""
    spec = importlib.util.spec_from_file_location(f"script_{name.split('_')[0]}", ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------------------------
# vccpipe.proxy
# --------------------------------------------------------------------------------------------------------------
def _small_h1(d, rng):
    """Three fake H1 files on a gene list that differs from the axis (100 axis genes missing, 50 extra genes).

    Targets: G0-G7 (planted 85% knockdown, except G1 which has none), X0 (not on the axis), and G9 in a third file
    with only 300 cells (too few). Controls: 300 in the first two files (shared), 20 in the third.
    """
    axis = np.array([f"G{i}" for i in range(1500)])
    h1_genes = np.concatenate([axis[:1400], [f"X{i}" for i in range(50)]])
    targets = [f"G{i}" for i in range(8)] + ["X0"]
    lfc, _ = fakes.planted_effects(targets, h1_genes, rng)
    lfc[1, 1] = 0.0                                            # G1: no knockdown → must be filtered out
    paths = [d / "h1_a.h5ad", d / "h1_b.h5ad", d / "h1_c.h5ad"]
    fakes.make_fake_h1(paths[:2], h1_genes, targets, rng, cells_per_target=420, n_ctrl=300, lfc=lfc)
    fakes.make_fake_h1(paths[2:], h1_genes, ["G9"], rng, cells_per_target=300, n_ctrl=20)
    return axis, paths


def test_proxy_format_and_filters():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        axis, paths = _small_h1(d, np.random.default_rng(0))
        out = d / "proxy" / "real.h5ad"
        kept = build_h1_proxy(paths, axis, out, n_ctrl=1000, kd_min=0.8, median_umi=20000, seed=3, block_rows=256)
        a = ad.read_h5ad(out)
        assert list(a.var_names) == list(axis), "output must be on the VCC axis, in its order"
        assert a.obs.index.is_unique
        counts = a.obs["target_gene"].astype(str).value_counts()
        assert counts[CONTROL] == 300, "min(n_ctrl, available controls), taken once from the first file"
        assert set(counts.index) - {CONTROL} == set(kept)
        assert all(counts[t] == 400 for t in kept), "exactly 400 cells per kept target"
        assert "G1" not in kept, "a target without knockdown must be filtered out"
        kd = pd.read_csv(out.with_suffix(".kd.csv"))
        assert set(kd["target"]) == {f"G{i}" for i in range(8)}, "candidates: on the axis and >= 400 cells"
        assert sorted(kd.loc[kd["kept"], "target"]) == kept
        assert (kd["passes"] == ((kd["ctrl_cpm"] > 5) & (kd["kd_frac"] >= 0.8))).all()
        clear = kd[(kd["ctrl_cpm"] > 50) & (kd["target"] != "G1")]
        assert len(clear) >= 2 and np.allclose(clear["kd_frac"], 0.85, atol=0.06), clear
        assert len(kept) >= 3
        X = sp.csr_matrix(a.X)
        assert X.data.min() >= 0 and not np.modf(X.data)[0].any(), "raw integer counts"
        assert X[:, 1400:].nnz == 0, "axis genes that H1 lacks must be all-zero columns"
        lib = np.asarray(X.sum(axis=1)).ravel()
        is_ctrl = (a.obs["target_gene"].astype(str) == CONTROL).to_numpy()
        assert abs(np.median(lib[is_ctrl]) / 20000 - 1) < 0.03, np.median(lib[is_ctrl])
        assert not (d / "proxy" / "real_chunks").exists(), "temporary chunks must be removed"


def test_proxy_counts_unchanged_without_thinning():
    """With p = 1 (median_umi above the data's depth) every proxy cell equals its source row on the shared genes."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        axis, paths = _small_h1(d, np.random.default_rng(1))
        out = d / "real.h5ad"
        build_h1_proxy(paths, axis, out, n_ctrl=100, kd_min=0.8, median_umi=10**9, seed=0, block_rows=100)
        a = ad.read_h5ad(out)
        X = sp.csr_matrix(a.X)
        src = [ad.read_h5ad(p) for p in paths]
        for i in np.random.default_rng(0).choice(a.n_obs, 60, replace=False):
            k, row = a.obs_names[i][3:].split("_")           # obs names are h1f<file>_<row>
            s = src[int(k)]
            assert s.obs["target_gene"].iloc[int(row)] == a.obs["target_gene"].iloc[i]
            mine = X[i].toarray().ravel()
            theirs = s[int(row)].to_df().iloc[0]
            shared = [g for g in axis if g in theirs.index]
            np.testing.assert_array_equal(mine[: len(shared)], theirs[shared].to_numpy())
            assert mine[len(shared):].sum() == 0


def test_proxy_reproducible_and_caps():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        axis, paths = _small_h1(d, np.random.default_rng(2))
        k1 = build_h1_proxy(paths, axis, d / "r1.h5ad", n_ctrl=200, seed=5)
        k2 = build_h1_proxy(paths, axis, d / "r2.h5ad", n_ctrl=200, seed=5)
        a1, a2 = ad.read_h5ad(d / "r1.h5ad"), ad.read_h5ad(d / "r2.h5ad")
        assert k1 == k2 and (a1.obs_names == a2.obs_names).all()
        assert (sp.csr_matrix(a1.X) != sp.csr_matrix(a2.X)).nnz == 0, "same seed → identical proxy"
        k3 = build_h1_proxy(paths, axis, d / "r3.h5ad", n_ctrl=200, seed=5, max_targets=2)
        assert len(k3) == 2 and set(k3) <= set(k1)
        k4 = build_h1_proxy(paths, axis, d / "r4.h5ad", n_ctrl=200, seed=5, targets=["G0", "G2", "G3", "X0"])
        assert set(k4) <= {"G0", "G2", "G3"}
        assert ad.read_h5ad(d / "r3.h5ad").n_obs == 200 + 2 * 400


# --------------------------------------------------------------------------------------------------------------
# configs/v1.yaml, the variant grid, the download planner
# --------------------------------------------------------------------------------------------------------------
def test_config_and_variants():
    from vccpipe.model import ModelParams
    cfg = yaml.safe_load(CONFIG.read_text())
    for key in ["paths", "data", "sources", "cis", "proxy", "model", "generator", "validation", "submission", "smoke"]:
        assert key in cfg, key
    ModelParams.from_json(cfg["model"])                         # every model key is a ModelParams field
    assert cfg["validation"]["folds"] == 2
    assert list(cfg["data"]["h1"]["files"])[0].startswith("test/"), "proxy controls come from the first H1 file"

    # The decisions of the v1 review, so a later edit cannot undo them silently.
    assert cfg["model"]["source_weights"]["h1_2025"] == 0.0, "v1 submits the validated K562-only model"
    assert cfg["model"]["template_sources"] == ["k562_gw"]
    assert cfg["validation"]["grid"]["z0"] == [2.5, 3.5, 4.5]
    assert cfg["validation"]["stage2"] == {"beta_scale": [1.0, 2.0, 3.0], "min_called_lfc": [0.0, 0.4]}
    assert cfg["proxy"]["max_strong_genes"] == 50 and cfg["proxy"]["strong_z"] == 4.5
    assert "release_32/gencode.v32.basic" in cfg["data"]["gencode"]["url"], "10x GRCh38-2020-A symbols = GENCODE v32"
    assert Path(cfg["data"]["gencode"]["url"]).name == Path(cfg["data"]["gencode"]["file"]).name

    v02 = _load_script("02_validate.py")
    full = v02.expand_variants(cfg)
    names = [v["name"] for v in full]
    assert len(names) == len(set(names)) == 3 * 2 * 3 + 2, names   # nb has no spread → its duplicates dropped
    assert all(v["spread"] == 0.0 for v in full if v["generator"] == "nb")
    abl = {v["name"]: v for v in full if v["kind"] == "ablation"}
    assert abl["abl_template_only"]["model"] == {"use_specific": False}
    assert abl["abl_no_cis"]["model"] == {"use_cis": False}
    p = v02.variant_params(cfg, abl["abl_no_cis"])
    assert p.use_cis is False and set(p.source_weights) <= {"k562_gw"}, "validation transfers from K562 only"
    assert p.template_sources == ["k562_gw"] and p.beta_scale == 1.0 and p.min_called_lfc == 0.0

    # Stage 2: around the best GRID variant of each generator (an ablation never becomes a centre), 3 x 2 - 1 each.
    rng = np.random.default_rng(0)
    res = pd.DataFrame([{"variant": v["name"], "kind": v["kind"], "generator": v["generator"],
                         "overall": rng.random()} for v in full])
    res.loc[res["kind"] == "ablation", "overall"] = 9.0
    st2 = v02.stage2_variants(cfg, full, res)
    assert len(st2) == 2 * 5, [v["name"] for v in st2]
    for gen in ("template", "nb"):
        g = res[(res["kind"] == "grid") & (res["generator"] == gen)].sort_values("overall").iloc[-1]["variant"]
        mine = [v for v in st2 if v["generator"] == gen]
        assert {v["parent"] for v in mine} == {g} and all(v["name"].startswith(g + "__") for v in mine)
        assert {(v["model"]["beta_scale"], v["model"]["min_called_lfc"]) for v in mine} == \
            {(1.0, 0.4), (2.0, 0.0), (2.0, 0.4), (3.0, 0.0), (3.0, 0.4)}, "(1, 0) is the winner itself"
        parent = next(v for v in full if v["name"] == g)
        assert all(v["spread"] == parent["spread"] for v in mine)
        assert all(v02.variant_params(cfg, v).z0 == v02.variant_params(cfg, parent).z0 for v in mine)
    assert v02.best_row(res, [v["name"] for v in full])["kind"] == "grid", "ablations never win"

    smoke = v02.load_config(CONFIG, smoke=True)
    assert len(v02.expand_variants(smoke)) == 2
    assert smoke["validation"]["stage2"] == {"beta_scale": [1.0, 2.0], "min_called_lfc": [0.0]}
    assert smoke["submission"]["cells_per_pert"] == 5
    assert smoke["paths"]["work_dir"] == (ROOT / "work" / "smoke").resolve(), "smoke runs write elsewhere"
    assert v02.load_config(CONFIG)["paths"]["work_dir"] == (ROOT / "work").resolve()


def _tiny_k562():
    """A 2-target K562-like ResponseSet on 10 genes with known z-scores (se = 0.1 everywhere it is measured).

    g0..g7 are testable (control 100 CPM), g8 is not measured, g9 is below the 5 CPM DE gate.
    Target g0 (its own gene is g0): strong on g0 (its knockdown: never counted), g1-g4, g8 (unmeasured), g9 (gate).
    Target X (not on the axis): strong on g1 only.
    """
    from vccpipe.sources import ResponseSet
    genes = np.array([f"g{i}" for i in range(10)])
    ctrl = np.full(10, 100.0)
    ctrl[9] = 1.0
    measured = np.ones(10, dtype=bool)
    measured[8] = False
    tc = np.tile(ctrl, (2, 1))
    tc[0, [0, 1, 2, 3, 4, 8, 9]] *= 0.25                    # log2 0.25 = -2 → |z| ≈ 20
    tc[1, 1] *= 4.0
    tc[:, ~measured] = 0.0
    se = np.where(measured, 0.1, np.inf) * np.ones((2, 1))
    meta = pd.DataFrame({"n_cells": [100, 100], "kd_remaining": [0.2, np.nan], "phenotype_p": [1e-4, 1e-4]},
                        index=["g0", "X"])
    return ResponseSet(name="k562_gw", genes=genes, measured=measured, targets=["g0", "X"], target_cpm=tc,
                       ctrl_cpm=np.where(measured, ctrl, 0.0), se_lfc=se, meta=meta)


def test_proxy_strength_filter():
    """01's filter: count strong, DE-testable, non-own genes per H1 target in K562; exclude above the cap."""
    v01 = _load_script("01_build_sources.py")
    k562 = _tiny_k562()
    counts = v01.strong_gene_counts(k562, ["Y", "X", "g0"], strong_z=4.5)
    assert counts.to_dict() == {"X": 1, "g0": 4}, counts          # Y: not in K562; order of the targets given
    table, allowed = v01.proxy_strength_filter(k562, ["g0", "X", "Y"], max_strong_genes=2, strong_z=4.5)
    assert allowed == ["X", "Y"], "g0 has 4 > 2 strong genes; Y (not measured in K562) is kept"
    assert table.loc["g0", "excluded"] and not table.loc["X", "excluded"] and not table.loc["Y", "excluded"]
    assert not table.loc["Y", "in_k562"] and np.isnan(table.loc["Y", "n_strong"])
    _, allowed = v01.proxy_strength_filter(k562, ["g0", "X", "Y"], max_strong_genes=4, strong_z=4.5)
    assert allowed == ["g0", "X", "Y"], "the cap is 'more than', like the VCC panel's max of 42 vs 50"
    table, allowed = v01.proxy_strength_filter(k562, ["g0", "X"], max_strong_genes=None, strong_z=4.5)
    assert allowed is None and not table["excluded"].any(), "null = filter off (no restriction)"
    assert v01.strong_gene_counts(k562, ["g0"], strong_z=25.0).to_dict() == {"g0": 0}


def test_validation_resume_bookkeeping():
    """02 --resume keeps only rows of this engine and proxy; a stale bundle or other columns rename the file."""
    v02 = _load_script("02_validate.py")
    cols = v02.result_columns()
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "results.csv"

        def write(rows):
            df = pd.DataFrame([{c: np.nan for c in cols} | r for r in rows], columns=cols)
            df.to_csv(csv, index=False)

        rows = [{"variant": "a", "engine": "cpu", "n_targets": 10, "proxy_id": "P1"},
                {"variant": "b", "engine": "gpu", "n_targets": 10, "proxy_id": "P1"},
                {"variant": "c", "engine": "cpu", "n_targets": 9, "proxy_id": "P1"},
                {"variant": "d", "engine": "cpu", "n_targets": 10, "proxy_id": "P0"}]
        write(rows)
        done = v02.prepare_results_csv(csv, resume=True, stale=False, engine="cpu", n_targets=10, proxy_id="P1")
        assert done == {"a"}
        assert pd.read_csv(csv)["variant"].tolist() == ["a"], "other engines / proxies are dropped from the file"
        backups = list(Path(d).glob("results.*.csv"))
        assert len(backups) == 1 and len(pd.read_csv(backups[0])) == 4, "the original is kept"
        for b in backups:
            b.unlink()

        assert v02.prepare_results_csv(csv, True, False, "cpu", 10, "P1") == {"a"}
        assert not list(Path(d).glob("results.*.csv")), "nothing to drop: no backup"

        assert v02.prepare_results_csv(csv, resume=True, stale=True, engine="cpu", n_targets=10,
                                       proxy_id="P1") == set()
        assert not csv.exists() and len(list(Path(d).glob("results.*.csv"))) == 1, "stale bundle → renamed"

        pd.DataFrame([{"variant": "a", "overall": 0.1}]).to_csv(csv, index=False)   # an older script's columns
        assert v02.prepare_results_csv(csv, True, False, "cpu", 10, "P1") == set() and not csv.exists()

        write(rows)
        assert v02.prepare_results_csv(csv, resume=False, stale=False, engine="cpu", n_targets=10,
                                       proxy_id="P1") == set() and not csv.exists(), "no --resume → renamed"


def test_submission_refuses_control_templates():
    """03 stops before loading anything when the parameter file uses the template generator (open rules
    question), unless --allow-control-templates is given."""
    from vccpipe.model import ModelParams
    with tempfile.TemporaryDirectory() as d:
        params = Path(d) / "best_params.json"
        params.write_text(json.dumps({"variant": "x", "generator": "template", "spread": 0.0,
                                      "model": json.loads(ModelParams().to_json())}))
        p = subprocess.run([str(PY), str(ROOT / "scripts" / "03_make_submission.py"), "--config", str(CONFIG),
                            "--params", str(params), "--controls-dir", d, "--dry-run"],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert p.returncode != 0, p.stdout + p.stderr
        assert "--allow-control-templates" in p.stderr and "best_params_nb.json" in p.stderr, p.stderr


def test_submission_maps_sources_to_another_axis():
    """03 moves a source onto a bundle axis with another gene order / genes (final phase) instead of exiting."""
    v03 = _load_script("03_make_submission.py")
    k562 = _tiny_k562()
    new_axis = np.array(["g3", "g0", "NEW", "g1", "g2", "g4", "g5", "g6", "g7", "g8"])   # g9 dropped, NEW added
    rs = v03.to_bundle_axis(k562, new_axis, "test bundle")
    assert list(rs.genes) == list(new_axis) and rs.targets == k562.targets
    old = {g: j for j, g in enumerate(k562.genes)}
    for j, g in enumerate(new_axis):
        if g == "NEW":
            assert not rs.measured[j] and (rs.target_cpm[:, j] == 0).all() and np.isinf(rs.se_lfc[:, j]).all()
        else:
            assert rs.measured[j] == k562.measured[old[g]]
            np.testing.assert_array_equal(rs.target_cpm[:, j], k562.target_cpm[:, old[g]])
    assert v03.to_bundle_axis(k562, k562.genes, "same axis") is k562, "same axis: no copy, no message"


def test_tsubame_pipeline_script_logic():
    """run_pipeline.sh with stub commands (no cluster): module init check, TMPDIR, which parameter file and flags
    step 03 gets, and that the job stops when `module` is missing."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d).resolve()                           # the shell reports the physical path (macOS: /private/var)
        proj, stubs = d / "proj with space", d / "stubs"
        (proj / ".venv" / "bin").mkdir(parents=True)
        (proj / "controls").mkdir()
        (proj / "controls" / "gene_names.csv").write_text("gene_name\nA\n")
        (proj / "controls_DEF").mkdir()
        (proj / "controls_DEF" / "gene_names.csv").write_text("gene_name\nA\n")
        stubs.mkdir()
        calls = d / "calls.log"
        # The fake python logs every call; `-c` snippets succeed, the yaml one prints the work folder.
        py = proj / ".venv" / "bin" / "python"
        py.write_text(f'#!/bin/sh\nif [ "$1" = -c ]; then case "$2" in *yaml*) echo work;; esac; exit 0; fi\n'
                      f'echo "$*" >> "{calls}"\n')
        for name, body in {"module": "exit 0", "vcc": 'echo "vcc 0.2.2"', "nvidia-smi": "echo GPU 0"}.items():
            (stubs / name).write_text(f"#!/bin/sh\n{body}\n")
        for f in [py, *stubs.iterdir()]:
            f.chmod(0o755)
        script = proj / "run_pipeline.sh"
        shutil.copy(ROOT / "tsubame" / "run_pipeline.sh", script)

        def job(**env):
            calls.unlink(missing_ok=True)
            e = {k: v for k, v in os.environ.items() if k not in ("T4TMPDIR", "JOB_ID")}
            e.update(PATH=f"{stubs}:/usr/bin:/bin", HOME=str(d), **env)
            p = subprocess.run(["sh", str(script)], cwd=proj, env=e, capture_output=True, text=True, timeout=60)
            return p, (calls.read_text().splitlines() if calls.exists() else [])

        p, c = job()
        assert p.returncode == 0, p.stdout + p.stderr
        assert [ln.split()[0] for ln in c] == ["scripts/01_build_sources.py", "scripts/02_validate.py",
                                               "scripts/03_make_submission.py"], c
        assert "--gpu" in c[1].split() and "--resume" not in c[1].split()
        assert "--params work/validation/best_params_nb.json" in c[2] and "--allow-control-templates" not in c[2]
        assert "vcc 0.2.2" in p.stdout and f"TMPDIR={proj}/work/tmp" in p.stdout
        assert (proj / "work" / "tmp").is_dir()

        p, c = job(ALLOW_CONTROL_TEMPLATES="1", SKIP_BUILD="1", RESUME="1", CONTROLS_DIR="controls_DEF",
                   DRY_RUN="1")
        assert p.returncode == 0, p.stdout + p.stderr
        assert [ln.split()[0] for ln in c] == ["scripts/02_validate.py", "scripts/03_make_submission.py"], c
        assert "--resume" in c[0].split()
        assert c[1].endswith("--params work/validation/best_params.json --allow-control-templates "
                             "--controls-dir controls_DEF --dry-run"), c[1]

        p, c = job(T4TMPDIR=str(d / "local"), PARAMS="work/validation/best_params_template.json")
        assert p.returncode == 0 and f"TMPDIR={d / 'local'}" in p.stdout
        assert "--params work/validation/best_params_template.json" in c[-1]

        (stubs / "module").unlink()                     # no module command and no init script that defines it
        if not Path("/etc/profile.d/modules.sh").exists():
            p, c = job()
            assert p.returncode != 0 and "module command missing" in p.stderr and not c, p.stdout + p.stderr


def test_download_script_lists_without_network():
    with tempfile.TemporaryDirectory() as d:
        cfg = yaml.safe_load(CONFIG.read_text())
        cfg["paths"]["data_dir"] = str(Path(d) / "data")
        (Path(d) / "c.yaml").write_text(yaml.safe_dump(cfg))
        p = subprocess.run(["bash", str(ROOT / "scripts" / "00_download_data.sh"), "--list", "--config",
                            str(Path(d) / "c.yaml")], capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        lines = [ln for ln in p.stdout.splitlines() if ln.strip().startswith(("replogle", "gencode", "h1"))]
        assert len(lines) == 3 + 1 + 3 and all("missing" in ln for ln in lines), p.stdout
        assert any("adata_Training.h5ad" in ln and "15.5 GB" in ln for ln in lines)


# --------------------------------------------------------------------------------------------------------------
# The whole pipeline on fake data
# --------------------------------------------------------------------------------------------------------------
def _fake_inputs(tmp):
    """Fake K562 bulk, two fake H1 files and a fake GTF on the REAL VCC gene axis; returns the H1 targets.

    The fake H1 responses copy the fake K562 responses (on K562's genes), so transfer has real signal to find.
    H1 targets are panel genes chosen to be clearly expressed in the fake H1 controls, so the proxy's knockdown
    filter (which needs control CPM > 5) can see their knockdown.
    """
    genes = fakes.vcc_genes(CONTROLS)
    panel = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str).tolist()
    gidx = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(2026)
    data = tmp / "data"
    for sub in ("replogle", "h1_2025", "annotation"):
        (data / sub).mkdir(parents=True, exist_ok=True)

    # make_fake_h1 draws its control profile first, from the rng it is given: a copy of that rng shows it to us.
    rng_h1 = np.random.default_rng(11)
    probe = copy.deepcopy(rng_h1).gamma(0.4, 1.0, genes.size) + 1e-3
    base_cpm = probe / probe.sum() * 1e6
    expressed = [t for t in panel if base_cpm[gidx[t]] > 30]
    h1_targets = sorted(rng.choice(expressed, 12, replace=False).tolist())

    others = [t for t in panel if t not in h1_targets]
    outside = [g for g in genes[rng.choice(genes.size, 200, replace=False)] if g not in set(panel)]
    k_targets = h1_targets + rng.choice(others, 48, replace=False).tolist() + outside[:10]
    filler = [g for g in genes[rng.permutation(genes.size)] if g not in set(k_targets)]
    k_set = set(k_targets) | set(filler[: 8000 - len(k_targets) - 1])
    k_genes = [g for g in genes if g in k_set] + [filler[8000]]     # the fake renames the LAST gene: a filler
    k = fakes.make_fake_replogle_bulk(data / "replogle" / "K562_gwps_raw_bulk_01.h5ad", np.array(k_genes),
                                      k_targets, rng)

    lfc = np.zeros((len(h1_targets), genes.size))
    kcol = np.array([gidx[g] for g in k_genes])
    for i, t in enumerate(h1_targets):
        lfc[i, kcol] = k["lfc"][k_targets.index(t)]
    h1 = fakes.make_fake_h1([data / "h1_2025" / "adata_A.h5ad", data / "h1_2025" / "adata_B.h5ad"], genes,
                            h1_targets, rng_h1, cells_per_target=420, n_ctrl=600, lfc=lfc)
    assert np.allclose(h1["base_cpm"], base_cpm), "tests/fakes.py changed its draw order: update the probe"
    fakes.make_fake_gtf(data / "annotation" / "fake.gtf.gz", genes, rng)
    return h1_targets


def _peak_child_gb():
    """Largest resident memory of any finished child process so far (bytes on macOS, kB on Linux)."""
    import resource
    r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def _run(args, log, expect_ok=True):
    t0 = time.time()
    p = subprocess.run([str(PY), *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=3600)
    log.write_text(p.stdout + "\n" + p.stderr)
    tail = "\n".join((p.stdout + p.stderr).splitlines()[-25:])
    print(f"\n--- {Path(args[0]).name} ({time.time() - t0:.0f} s, exit {p.returncode}, peak child RSS so far "
          f"{_peak_child_gb():.1f} GB) ---\n{tail}", flush=True)
    if expect_ok:
        assert p.returncode == 0, f"{args[0]} failed (exit {p.returncode}); log: {log}"
        return p.stdout
    assert p.returncode != 0, f"{args[0]} was expected to fail; log: {log}"
    return p.stdout + p.stderr


def _pipeline_config(tmp, **proxy):
    """A config like configs/v1.yaml with every path in the temporary folder (the real controls bundle aside), the
    smoke section set for the fake data, and `proxy` overrides for the smoke section. Returns (config path,
    smoke processed folder, smoke work folder)."""
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["paths"] = {"controls_dir": str(CONTROLS), "data_dir": str(tmp / "data"),
                    "processed_dir": str(tmp / "processed"), "work_dir": str(tmp / "work")}
    cfg["data"]["gencode"]["file"] = "annotation/fake.gtf.gz"
    cfg["data"]["h1"]["files"] = {"fake/adata_A.h5ad": None, "fake/adata_B.h5ad": None}
    smoke = cfg["smoke"]
    smoke["paths"] = {"processed_dir": str(tmp / "smoke" / "processed"), "work_dir": str(tmp / "smoke" / "work")}
    smoke["proxy"].update(n_ctrl=600, max_targets=12, **proxy)
    smoke["validation"].update(folds=2, ablations=[],
                               grid={"z0": [2.5], "null_p": [1.0], "generator": ["template", "nb"],
                                     "spread": [0.0]},
                               stage2={"beta_scale": [1.0, 2.0], "min_called_lfc": [0.0]})   # +1 per generator
    smoke["submission"]["cells_per_pert"] = 5
    conf = tmp / "config.yaml"
    conf.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return conf, tmp / "smoke" / "processed", tmp / "smoke" / "work"


def test_build_sources_strength_filter_reaches_proxy():
    """01 on fake data with a low strength cap: the strongest H1 targets are left out of the proxy it builds.

    The fake K562 responses have <= 40 changed genes per target (~0-20 of them at |z| > 4.5), so the real cap of
    50 excludes nothing; a cap of 18 excludes the strongest one or two. (This is separate from the end-to-end test
    because the scorer refuses a baseline that is degenerate on the smaller fake panel.)
    """
    if not (CONTROLS / "gene_names.csv").exists():
        _skip("no controls bundle in controls/")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        h1_targets = _fake_inputs(tmp)
        conf, _, work = _pipeline_config(tmp, max_strong_genes=18)
        out01 = _run([ROOT / "scripts" / "01_build_sources.py", "--config", conf, "--smoke"], tmp / "log01.txt")
        kept = (work / "h1_proxy" / "targets.txt").read_text().split()
        assert "strength filter:" in out01 and "excluded:" in out01
        sf = json.loads((work / "h1_proxy" / "strength_filter.json").read_text())
        assert sf["max_strong_genes"] == 18 and sf["n_h1_targets"] == len(h1_targets)
        strength = pd.read_csv(work / "h1_proxy" / "strength.csv", index_col=0)
        assert set(strength.index) == set(h1_targets) and strength["in_k562"].all()
        assert sorted(sf["excluded"]) == sorted(strength.index[strength["n_strong"] > 18]) and sf["n_excluded"] >= 1
        assert not set(sf["excluded"]) & set(kept), "excluded targets must not be in the proxy"
        assert set(kept) == set(h1_targets) - set(sf["excluded"]), "every other fake target passes the kd filter"
        assert (strength["in_proxy"] == strength.index.isin(kept)).all()
        real = ad.read_h5ad(work / "h1_proxy" / "real.h5ad", backed="r")
        assert set(real.obs["target_gene"].astype(str)) - {CONTROL} == set(kept)
        real.file.close()

        # --reuse with a proxy that the filter would now cut: a loud warning, not a silent reuse.
        conf2, _, _ = _pipeline_config(tmp, max_strong_genes=5)
        out = _run([ROOT / "scripts" / "01_build_sources.py", "--config", conf2, "--smoke", "--reuse"],
                   tmp / "log01_reuse.txt")
        assert "WARNING: the reused proxy contains" in out


def test_pipeline_end_to_end():
    if not (CONTROLS / "gene_names.csv").exists():
        _skip("no controls bundle in controls/")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        h1_targets = _fake_inputs(tmp)
        conf, proc, work = _pipeline_config(tmp)            # the real strength cap (50): nothing excluded here

        # ---- 01: sources, TSS, cis model, proxy ----
        out01 = _run([ROOT / "scripts" / "01_build_sources.py", "--config", conf, "--smoke"], tmp / "log01.txt")
        for f in ["k562_gw.npz", "h1_2025.npz", "tss.csv", "cis.json"]:
            assert (proc / f).exists(), f
        assert not (tmp / "processed").exists(), "--smoke must write to the smoke paths"
        kept = (work / "h1_proxy" / "targets.txt").read_text().split()
        assert 6 <= len(kept) <= 12 and set(kept) <= set(h1_targets), kept
        # The strength filter ran and is recorded next to the proxy (its effect: test_build_sources_strength_filter_
        # reaches_proxy; its logic: test_proxy_strength_filter).
        assert "strength filter:" in out01
        sf = json.loads((work / "h1_proxy" / "strength_filter.json").read_text())
        assert sf["max_strong_genes"] == 50 and sf["n_h1_targets"] == len(h1_targets) and sf["n_excluded"] == 0
        real = ad.read_h5ad(work / "h1_proxy" / "real.h5ad", backed="r")
        assert real.n_obs == 600 + 400 * len(kept) and real.n_vars == 18533
        real.file.close()

        # ---- 02: validation on the proxy with the official scorer (CPU engine), stage 1 then stage 2 ----
        out02 = _run([ROOT / "scripts" / "02_validate.py", "--config", conf, "--smoke"], tmp / "log02.txt")
        vdir = work / "validation"
        res = pd.read_csv(vdir / "results.csv")
        stage1 = ["z0-2.5_null_p-1_template_s0", "z0-2.5_null_p-1_nb"]
        stage2 = [f"{v}__beta_scale-2_min_called_lfc-0" for v in stage1]
        assert sorted(res["variant"]) == sorted(["control_paste", *stage1, *stage2]), res["variant"].tolist()
        assert sorted(res.loc[res["kind"] == "stage2", "parent"]) == sorted(stage1)
        assert (res.loc[res["kind"] == "stage2", "beta_scale"] == 2.0).all()
        assert res["proxy_id"].nunique() == 1 and (res["engine"] == "cpu").all()
        scores = ["overall", "pds", "mse", "nmae", "fid", "reach", "jac"]
        assert np.isfinite(res[scores].to_numpy(dtype=float)).all()
        fitted = res[res["kind"].isin(["grid", "stage2"])]
        assert np.isfinite(fitted[["beta_mean", "gamma_mean"]].to_numpy(dtype=float)).all()
        for v in stage1:      # the gains are fitted before beta_scale applies, so stage 2 refits the same gains
            a, b = (fitted.set_index("variant").loc[[v, f"{v}__beta_scale-2_min_called_lfc-0"], "gains_per_fold"])
            assert a == b, (v, a, b)
        assert not list(vdir.glob("pred_*.h5ad")), "prediction files must be deleted after scoring"
        from vccpipe.model import ModelParams
        best = json.loads((vdir / "best_params.json").read_text())
        per_gen = {g: json.loads((vdir / f"best_params_{g}.json").read_text()) for g in ("template", "nb")}
        for g, b in per_gen.items():
            assert b["generator"] == g and b["variant"] in set(fitted.loc[fitted["generator"] == g, "variant"])
            row = fitted.set_index("variant").loc[b["variant"]]
            assert row["overall"] == fitted.loc[fitted["generator"] == g, "overall"].max()
            assert b["cv_scores"]["overall"] == row["overall"]
        assert best["variant"] in set(fitted["variant"])
        assert best["cv_scores"]["overall"] == fitted["overall"].max()
        assert best == {**per_gen[best["generator"]], "created_utc": best["created_utc"]}
        for b in (best, *per_gen.values()):
            p = ModelParams.from_json(b["model"])
            assert np.isfinite([p.beta, p.gamma]).all() and p.beta == b["fitted_gains"]["beta"]
            assert p.template_sources == ["k562_gw"] and b["validation_sources"] == ["k562_gw"]
            assert p.source_weights == {"k562_gw": 1.0, "h1_2025": 0.0}, "the config's weights, for step 3"
            assert b["proxy"]["n_targets"] == len(kept) and b["proxy"]["excluded_strong"] == sf["n_excluded"]
        assert "Ranked by overall" in out02 and "stage 2:" in out02

        # ---- 03: the template generator needs --allow-control-templates (open rules question) ----
        refused = _run([ROOT / "scripts" / "03_make_submission.py", "--config", conf, "--smoke", "--dry-run",
                        "--controls-dir", CONTROLS, "--params", vdir / "best_params_template.json"],
                       tmp / "log03_refused.txt", expect_ok=False)
        assert "--allow-control-templates" in refused and "best_params_nb.json" in refused

        # ---- 03: submission against the real controls bundle, 5 cells per target, dry run ----
        # The default parameter file is the overall winner: allowed only with the flag if it uses templates.
        flag = ["--allow-control-templates"] if best["generator"] == "template" else []
        out03 = _run([ROOT / "scripts" / "03_make_submission.py", "--config", conf, "--smoke", "--dry-run",
                      "--controls-dir", CONTROLS, *flag], tmp / "log03.txt")
        sub = work / f"submission_{Path(CONTROLS).name}" / "submission.h5ad"   # one folder per controls bundle
        assert sub.exists() and not sub.with_suffix(".vcc").exists(), "smoke is a dry run: no .vcc"
        assert "vcc prep dry run: PASSED" in out03
        assert "vcc submit" not in out03, "a dry run must not even print a submit command"
        assert "h1_2025: weight 0" in out03, "v1 submits the validated K562-only model"
        assert sum("the template alone moves" in ln for ln in out03.splitlines()) == 3, "one check per context"
        s = ad.read_h5ad(sub, backed="r")
        assert s.n_obs == 3 * 300 * 5 and s.n_vars == 18533
        assert set(s.obs["context"].astype(str)) == {"A", "B", "C"}
        assert CONTROL not in set(s.obs["target_gene"].astype(str))
        s.file.close()
        prov = json.loads(sub.with_suffix(".json").read_text())          # provenance file next to the submission
        assert prov["sources"] == ["k562_gw"] and prov["template_sources"] == ["k562_gw"]
        assert prov["generator"] == best["generator"] and prov["variant"] == best["variant"]
        assert prov["allow_control_templates"] == bool(flag)
        assert len(prov["template_check"]) == 3


# --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    wanted = sys.argv[1:]
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)
             and (not wanted or n in wanted)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f} s)", flush=True)
        except Skipped as e:
            print(f"SKIP  {name}: {e}", flush=True)
        except Exception:
            failed += 1
            traceback.print_exc()
            print(f"FAIL  {name}  ({time.time() - t0:.1f} s)", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
