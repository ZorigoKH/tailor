---
title: unbundle
tags: open source, python, finance, statistics, testing
---

# unbundle, separating alpha from factor exposure

Built unbundle on top of sandwich: it splits a fund's monthly returns into the factor exposure anyone can buy for a few basis points and the alpha left over, with Newey-West standard errors and GRS and HAC Wald tests of whether the alphas are jointly zero.

Ships a parser and a one-week cache for the Kenneth R. French data library, rolling-window loadings, and a one-file HTML tearsheet with hover readouts that opens offline with no server.

57 tests, and every number quoted in the README is recomputed by the test suite, so the documentation cannot drift from the code. Continuous integration (CI) runs the linter and the tests on Python 3.10 through 3.13.
