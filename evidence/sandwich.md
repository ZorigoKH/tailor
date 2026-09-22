---
title: sandwich
tags: open source, python, numpy, econometrics, testing
---

# sandwich, an econometrics engine in NumPy

Wrote an econometrics engine in NumPy: ordinary least squares through a thin QR decomposition, HC0 to HC3, one- and two-way cluster-robust and Newey-West HAC standard errors, instrumental variables and two-stage least squares with first-stage F, Wu-Hausman, Sargan and Hansen J diagnostics, panel fixed effects and first differences.

51 tests check every number against statsmodels and linearmodels to a relative tolerance of 1e-9, so the tests are the specification rather than a smoke screen.
