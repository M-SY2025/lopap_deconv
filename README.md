# lopap_deconv

Deconvolution of LOPAP (Long Path Absorption Photometer) time series using a dual inverse-Gaussian instrument response function (IRF).

## Overview

LOPAP instruments measure trace-gas concentrations (e.g. HONO) via liquid-phase absorption. The liquid transport introduces a time delay and dispersion that smooth rapid atmospheric changes. This package estimates the IRF from step-response calibration data and applies nonnegative deconvolution to recover the original atmospheric signal.

**Key features:**

- Dual inverse-Gaussian IRF model (fast + slow dispersion paths)
- Automatic transport-delay estimation from step events
- Latin Hypercube search + L-BFGS-B optimisation for IRF fitting
- TV-regularised nonnegative deconvolution via IRLS
- Handles large time series with automatic downsampling

## Installation

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Quick start

1. **Prepare a training manifest** (CSV or Excel) listing your step-response LOPAP files:

   | file | t0 | direction |
   |---|---|---|
   | exp1.csv | 2024/8/7 16:54:45 | rise |
   | exp1.csv | 2024/8/7 17:30:00 | fall |

   Each LOPAP data file has two columns: datetime and concentration.

2. **Run the example workflow:**

   ```bash
   cd examples
   python run_deconv.py
   ```

   Edit the USER SETTINGS section at the top of `run_deconv.py` to point to your data files.

3. **Or use the Python API directly:**

   ```python
   from lopap_deconv import (
       read_lopap_manifest, cfg_default, pack_lopap_only,
       estimate_tau_from_lopap, fit_irf_sd_train,
       run_fixed_irf_recon, irf_from_theta,
   )

   # Load training data
   exps = read_lopap_manifest('manifest.csv')

   # Estimate transport delay
   tau, diag = estimate_tau_from_lopap(exps[0]['inst1'], exps[0]['events'])

   # Pack and train
   packs = [pack_lopap_only(e['inst1'], tau, e['events'], Kbase=5, mode='train')
            for e in exps]
   cfg = cfg_default()
   # ... configure bounds, regularisation, etc.
   sd_star, fval, detail = fit_irf_sd_train(packs, cfg)

   # Deconvolve
   packs_test = [pack_lopap_only(e['inst1'], tau, mode='deconv') for e in exps]
   res = run_fixed_irf_recon(packs_test, sd_star, cfg)
   ```

## Package structure

```
lopap_deconv/
    __init__.py       # public API
    io.py             # data file readers
    preprocess.py     # config, packing, tau estimation
    train.py          # IRF training (LHS + optimisation)
    deconvolve.py     # deconvolution solver + IRF builder
    plotting.py       # diagnostic plots
examples/
    run_deconv.py     # end-to-end workflow script
```

## Requirements

- Python >= 3.9
- NumPy >= 1.22
- SciPy >= 1.9
- pandas >= 1.4
- matplotlib >= 3.5
- openpyxl >= 3.0

## License

MIT
