# Numerical Optimization vs Machine Learning in Solubility Space

A Streamlit application for **numerical reconstruction of solubility spaces** using **probabilistic objective functions and numerical optimization**, combined with **calibrated machine-learning classifiers** and **geometric shell-sphere extraction**.

---

## ⚠️ Research use only

This application is intended **exclusively for research, methodological analysis, and visualization**.

It is **not** a predictive device and must **not** be used as a standalone decision-making tool for formulation, solvent selection, or industrial design.

---

## Conceptual clarification (important)

This work **does not apply the classical Hansen Solubility Parameter (HSP) method**.

Instead, it performs:

> **Numerical optimization of a solubility region in a Hansen-like descriptor space**  
> using **data-driven, probabilistic objective functions**.

The Hansen distance formulation is used **only as a geometric descriptor**.  
The solubility region (center and radius), the decision boundary, and the probabilistic interpretation are obtained **entirely by numerical optimization and statistical learning**, not by classical Hansen rules or tabulated parameters.

---

## Overview

The app reconstructs solubility spaces from experimental data labeled as:
- soluble,
- partially soluble,
- insoluble.

The solubility region is defined by:
- a center in (δd, δp, δh) space, and
- an effective radius,

obtained by **minimizing probabilistic loss functions**, not by enforcing hard geometric constraints.

The numerically optimized solubility space is compared with:
- calibrated machine-learning classifiers trained on the same descriptors, and
- ML-derived geometric shell-spheres extracted from isoprobability surfaces (*p ≈ iso*).

The comparison focuses on **geometric consistency and probabilistic agreement**, not on replacing physical interpretation with black-box prediction.

---

## Key features

- **Numerical optimization of solubility regions**
  - Objective functions:
    - GEOM (geometric penalty)
    - LOGLOSS (probabilistic cross-entropy)
    - BRIER (quadratic probability error)
    - HINGE (margin-based loss)
    - SOFTCOUNT (soft classification error)
  - Local and global optimizers:
    - Powell, L-BFGS-B, TNC
    - Differential Evolution, Dual Annealing, SHGO
    - Genetic Algorithm
    - Reparameterized unconstrained optimization
- Explicit **probabilistic mapping** from normalized distance (RED) to solubility probability
- Supervised ML classifiers with **probability calibration**
- Automatic ML model selection based on cross-validation
- Group-wise cross-validation (LOGO) with automatic fallback to Leave-One-Out (LOO)
- Reconstruction of **ML shell-spheres** from isoprobability surfaces
- Interactive 3D visualization and publication-ready 2D plots
- Full export of results to a multi-sheet **Excel file**

---

## Repository structure

