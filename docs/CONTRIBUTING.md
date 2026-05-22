# Contributing

## Development setup

```bash
git clone https://github.com/DjamelZair/CPM-parameter-Posterior-Estimate-Generator.git
cd CPM-parameter-Posterior-Estimate-Generator
pip install -e ".[dev]"
pytest
```

## Repository layout

```
cll_cpm_inversion/
    __init__.py     re-exports the top-level API
    features.py     mask -> 5-feature row
    invert.py       Sobol-weighted k-NN matcher + posterior summary
    pipeline.py     orchestrator (infer_from_masks / infer_from_features)
    cli.py          `cll-invert` command-line entry point
    data/           bundled synthetic library, Sobol indices, LOO benchmark
    tests/          unit + integration tests
examples/
    01_quickstart.ipynb
    02_features_only.ipynb
docs/
    METHODS.md      scientific summary
    PARAMETERS.md   CPM parameter reference
    CONTRIBUTING.md this file
```

## Where to land changes

| Change                                                  | File                |
|---------------------------------------------------------|---------------------|
| Add a new morphology feature                            | `features.py` + bundled library regeneration |
| Tweak the matcher metric or k                           | `invert.py`         |
| Add a new identifiability threshold                     | `invert.py` `IDENTIFIABILITY_THRESHOLDS` |
| Extend the bundled synthetic library                    | `data/` (separate script) |
| New end-to-end entry point                              | `pipeline.py` + `cli.py` |

## Running tests

```bash
pytest cll_cpm_inversion/tests -q
```

Tests cover feature extraction (synthetic disks with known geometry)
and matcher round-trip identity (feeding a library sample back in must
return that sample at rank 0).

## Regenerating the bundled library

The synthetic library, Sobol indices, and LOO benchmark are produced
upstream of this repository (the CPM simulation + Saltelli sweep). If
you have re-run the upstream simulation:

1. Build the joined sample-level table (one row per Saltelli sample,
   PARAMS + features); save to `data/synthetic_library.csv`.
2. Run SALib on (surrogate of) the table to produce per-(feature,
   parameter) `S1` and `ST`; save to `data/sobol_indices.csv`.
3. Run a Sobol-weighted k-NN LOO with k = 20 against the new library
   and save per-parameter R^2 and Pearson r to
   `data/identifiability_loo.csv`.

Each step has a reference implementation in the thesis repository.
