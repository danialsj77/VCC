"""vccpipe: our Virtual Cell Challenge 2026 pipeline.

Modules
-------
common      shared constants, paths and expression helpers (CPM, means, variances)
generator   turn predicted mean expression into realistic integer-count cells
harness     score predictions with Arc's official cell-eval2 scorer on a proxy with known answers
submission  build, check and package a VCC submission file
sources     public perturbation screens as per-target response matrices on the VCC gene axis
cis         genome neighbourhood of each target (CRISPRi silences genes sharing its promoter)
model       the effect model: predicted mean expression per (context, target, gene)

Notebook 01 explains the harness and generator step by step; this package holds the production versions.
"""

__version__ = "0.1.0"
