#!/usr/bin/env python3
"""
run_deconv.py — LOPAP IRF estimation & deconvolution workflow.

Three-step workflow (see SI Text S4):
  Step 1: Load training data with step-response events
  Step 2: Estimate IRF (dual inverse Gaussian kernel)
  Step 3: Deconvolve any LOPAP measurement series

Required inputs:
  1. TRAIN_MANIFEST: CSV/XLSX listing training LOPAP files with step events.
       file,          t0,                  direction
       exp1.csv,      2024/8/7 16:54:45,   rise
       exp1.csv,      2024/8/7 17:30:00,   fall

  2. TEST_MANIFEST (optional): either a single LOPAP data file
     (e.g. 'field_day1.xlsx') or a CSV list of multiple files:
       file
       field_day1.xlsx
       field_day2.xlsx
     If empty, deconvolution runs on the training data itself.

Each LOPAP data file: Column 1 = datetime, Column 2 = concentration.
Data should include sufficient baseline BEFORE the first t0.
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import tkinter as tk
from tkinter import filedialog

from lopap_deconv import (
    read_lopap_manifest, read_lopap_raw, read_lopap_test_list,
    cfg_default, pack_lopap_only, estimate_tau_from_lopap,
    fit_irf_sd_train, run_fixed_irf_recon,
    plot_irf_diagnostics, plot_recon_diagnostics,
)


def _pick_file(title, optional=False):
    """Open a file dialog and return the selected path."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[('Excel/CSV', '*.xlsx *.xls *.csv'), ('All files', '*.*')],
    )
    root.destroy()
    if not path and not optional:
        raise SystemExit('No file selected — exiting.')
    return path


# =====================================================================
#  USER SETTINGS — edit this section
# =====================================================================

# ---- Data paths ----
# Set USE_FILE_DIALOG = True  to pick files via dialog each time.
# Set USE_FILE_DIALOG = False to use the hardcoded paths below.
USE_FILE_DIALOG = True

TRAIN_MANIFEST = ''   # training manifest (ignored when USE_FILE_DIALOG=True)
TEST_MANIFEST  = ''   # test data (ignored when USE_FILE_DIALOG=True)

# ---- Run control ----
RUN_DECONV      = True
PLOT_TRAIN_DIAG = True
PLOT_RECON_DIAG = True

# ---- Signal processing ----
SMOOTH_WIN  = 1        # movmean window for LOPAP signal (seconds)
T_WINDOW    = 3000     # IRF time window (seconds, > expected tmax)
ALPHA_ONSET = 0.05     # detectable onset threshold (fraction of step)
KBASE       = 5        # baseline correction: number of initial points

# ---- Delay search range ----
DELAY_MARGIN = 150     # seconds; search half-width around tau_global

# ---- IRF search bounds (dual-IG: [tau_f, Pe_f, w, tau_s, Pe_s]) ----
LB_OVERRIDE = None     # e.g. [250, 30, 0.4, 250, 5]
UB_OVERRIDE = None     # e.g. [350, 35, 0.65, 300, 20]

# ---- Deconvolution options ----
MAX_PTS_DECONV = 20000
DT_DECONV      = 0     # output grid: 0=native sampling, 1=1s, N=Ns
KBASE_DECONV   = 0     # 0 = data as-is; >0 = subtract mean of first N pts

# ---- Tau estimation options ----
OPTS_TAU = {
    'smooth_win':    21,
    'search_max':    2500,
    'alpha':         ALPHA_ONSET,
    'beta':          0.20,
    'onset_persist': 10,
    'confirm_win':   300,
    'preA':          600,
    'preB':          100,
    'tau_min':       100,
    'tau_max':       900,
}

# =====================================================================
#  END OF USER SETTINGS
# =====================================================================


def smooth(x, win):
    """Moving-mean smoothing (equivalent to MATLAB smoothdata(...,'movmean'))."""
    if win <= 1:
        return x.copy()
    return uniform_filter1d(x, size=win, mode='nearest')


def main():
    global TRAIN_MANIFEST, TEST_MANIFEST
    import matplotlib.pyplot as plt

    # ---- file dialog ----
    if USE_FILE_DIALOG:
        TRAIN_MANIFEST = _pick_file('Select TRAINING manifest (CSV/Excel)')
        TEST_MANIFEST  = _pick_file('Select TEST data file (cancel = use training data)', optional=True)

    # ========== A) Load training data ==========
    print('\n=== A) LOADING TRAINING DATA ===')
    exps_train = read_lopap_manifest(TRAIN_MANIFEST)
    n_train = len(exps_train)
    print(f'Loaded {n_train} training experiments from manifest.')

    # ========== B) Estimate transport delay tau ==========
    print('\n=== B) ESTIMATING TRANSPORT DELAY (tau) ===')
    for i, exp in enumerate(exps_train):
        inst1_s = smooth(exp['inst1'], SMOOTH_WIN)
        tau_i, _ = estimate_tau_from_lopap(inst1_s, exp['events'], OPTS_TAU)
        exp['tau_est'] = tau_i
        print(f'  Exp {i}: tau_est = {tau_i} s')

    tau_all = [e['tau_est'] for e in exps_train]
    tau_global = int(round(np.median(tau_all)))
    print(f'=== Global tau (median) = {tau_global} s ===')

    # ========== C) Pack data for training ==========
    print('\n=== C) PACKING DATA FOR TRAINING ===')
    packs_train = []
    for exp in exps_train:
        inst1_s = smooth(exp['inst1'], SMOOTH_WIN)
        pack = pack_lopap_only(inst1_s, tau_global, exp['events'], KBASE, 'train')
        packs_train.append(pack)
    print(f'Packed {n_train} training experiments.')

    # ========== D) Train IRF ==========
    print('\n=== D) TRAINING IRF (dual inverse Gaussian) ===')
    cfg = cfg_default()
    cfg['t_window']    = T_WINDOW
    cfg['alpha_onset'] = ALPHA_ONSET
    cfg['Kbase']       = KBASE

    cfg['train'] = {
        'lopap_only': True,
        'tau_known':  tau_global,
        'tm': {
            'tmax_cap':   T_WINDOW,
            'search_max': T_WINDOW,
        },
        'search_N_coarse':      8000,
        'search_N_topk':        50,
        'search_auto_tighten':  True,
        'auto_search':          True,
        'use_isolated_events':  True,
        'do_select_topk':       True,
        'two_pass':             True,
    }

    cfg['reg']['lambda1_smooth'] = 0.001
    cfg['reg']['lambda2_curv']   = 0.001
    cfg['reg']['win'] = {
        'guard_pre':       250,
        'guard_post':      60,
        'min_len':         200,
        'disable_mid_clip': True,
        'pre_need':        200,
    }

    cfg['opt']['fmincon'] = {
        'maxiter': 200,
        'maxfev':  2000,
        'ftol':    1e-6,
        'gtol':    1e-8,
    }

    # ---- Bounds ----
    if LB_OVERRIDE is not None and UB_OVERRIDE is not None:
        cfg['train']['lb'] = [tau_global - DELAY_MARGIN] + LB_OVERRIDE
        cfg['train']['ub'] = [tau_global + DELAY_MARGIN] + UB_OVERRIDE
    else:
        cfg['train']['lb'] = [tau_global - DELAY_MARGIN, 250, 30, 0.4, 250, 5]
        cfg['train']['ub'] = [tau_global + DELAY_MARGIN, 350, 35, 0.65, 300, 20]
    cfg['train']['search_lb'] = cfg['train']['lb'][1:]
    cfg['train']['search_ub'] = cfg['train']['ub'][1:]

    sd_star, fval, detail = fit_irf_sd_train(packs_train, cfg)

    # ---- Print results ----
    theta_str = np.array2string(np.round(sd_star['theta'], 2), separator=', ')
    print(f'\n========== TRAIN RESULTS ==========')
    print(f'  theta    = {theta_str}')
    print(f'  td (detectable delay) = {round(sd_star["theta"][0])} s')
    print(f'  tmax (last arrival)   = {sd_star["tmax"]} s')
    print(f'  Tmem (memory time)    = {sd_star["tmax"] - round(sd_star["theta"][0])} s')
    print(f'  fval = {fval:.6g}')
    print(f'===================================')

    # ---- Training diagnostics ----
    if PLOT_TRAIN_DIAG:
        plot_irf_diagnostics(sd_star, packs_train, detail, cfg)
        plt.show(block=False)

    # ========== E) Deconvolution ==========
    if not RUN_DECONV:
        print('\n>>> RUN_DECONV=False, skipping deconvolution.')
        return

    print('\n=== E) DECONVOLUTION (fixed IRF) ===')

    # ---- Load test data ----
    if TEST_MANIFEST:
        ext_test = os.path.splitext(TEST_MANIFEST)[1].lower()
        if ext_test in ('.xls', '.xlsx'):
            print(f'Loading single test file: {TEST_MANIFEST}')
            raw_test = read_lopap_raw(TEST_MANIFEST, DT_DECONV)
            exps_test = [{
                'filepath':     TEST_MANIFEST,
                'inst1':        raw_test['conc'].copy(),
                't_sec':        raw_test['t_sec'].copy(),
                't0_datetime':  raw_test['t0_datetime'],
                'dt_out':       raw_test['dt_out'],
                'raw':          raw_test,
            }]
        else:
            print(f'Loading test data from list: {TEST_MANIFEST}')
            exps_test = read_lopap_test_list(TEST_MANIFEST, DT_DECONV)
    else:
        print('No TEST_MANIFEST specified. Using training data for deconvolution.')
        exps_test = exps_train

    n_test = len(exps_test)

    # ---- Deconv config ----
    tau_deconv = round(sd_star['theta'][0])
    print(f'Using trained delay for deconv: tau_deconv = {tau_deconv} s')

    cfg_deconv = cfg.copy()
    cfg_deconv['reg'] = cfg['reg'].copy()
    cfg_deconv['reg']['prepad']      = tau_deconv
    cfg_deconv['reg']['tailpad']     = tau_deconv
    cfg_deconv['reg']['lambda_end0'] = 0
    cfg_deconv['skip_diagnostics']   = True
    cfg_deconv['do_refine']          = False
    cfg_deconv['tau_init']           = tau_deconv
    cfg_deconv['tmax_trained']       = sd_star['tmax']

    dt_eff = exps_test[0].get('dt_out', 1)
    if dt_eff < 1:
        dt_eff = 1
    print(f'  Test data time step: dt_eff = {dt_eff} s')

    cfg_deconv['dt_eff'] = dt_eff
    cfg_deconv['reg']['lambda_tv']     = 0.009
    cfg_deconv['reg']['tv_eps_frac']   = 0.05
    cfg_deconv['reg']['max_points']    = 8000   # auto-downsample if n > this
    cfg_deconv['reg']['tv_maxit']      = 4      # IRLS iterations (default 8)

    # ---- Pack test data ----
    packs_test = []
    for exp in exps_test:
        win = max(3, round(SMOOTH_WIN / dt_eff))
        inst1_s = smooth(exp['inst1'], win)
        tau_grid  = round(tau_deconv / dt_eff)
        tmax_grid = round(T_WINDOW / dt_eff) + round(sd_star['theta'][0] / dt_eff)
        pack = pack_lopap_only(inst1_s, tau_grid, None, KBASE_DECONV, 'deconv', tmax_grid)
        pack['dt_eff'] = dt_eff
        packs_test.append(pack)

    cfg_deconv['prepad_grid'] = tmax_grid

    # ---- Warmup note ----
    tmax_trained = sd_star['tmax']

    # ---- Run deconvolution ----
    res = run_fixed_irf_recon(packs_test, sd_star, cfg_deconv)

    print(f'\n========== DECONVOLUTION RESULTS ==========')
    for i in range(n_test):
        print(f'  Exp {i}: R²(reconvolution) = {res["r2_inst1"][i]:.4f}')
    print(f'=============================================')

    # ---- Diagnostics ----
    if PLOT_RECON_DIAG:
        plot_recon_diagnostics(res, packs_test, cfg_deconv)
        plt.show(block=False)

    # ========== F) Export ==========
    print('\n=== F) EXPORT ===')
    for i in range(n_test):
        pack  = packs_test[i]
        c_hat  = res['c_hat'][i]
        y1_hat = res['y1_hat'][i]
        n0 = pack['n']
        dt = dt_eff
        y1 = pack['y1'][:n0]

        c_col  = np.full(n0, np.nan)
        c_col[:len(c_hat)] = c_hat
        y1h_col = np.full(n0, np.nan)
        y1h_col[:len(y1_hat)] = y1_hat

        prepad = pack.get('prepad', pack.get('prepad_base', 0))
        n_raw = pack.get('n_raw', 0)
        tailpad_start = prepad + n_raw

        if KBASE_DECONV == 0:
            warmup_idx = round(tmax_trained / dt)
        else:
            warmup_idx = prepad

        idx_lo = max(warmup_idx, prepad)
        idx_hi = tailpad_start if tailpad_start > 0 else n0

        idx_pack = np.arange(idx_lo, idx_hi)
        idx_raw  = idx_pack - prepad
        t0_dt    = exps_test[i]['t0_datetime']
        datetime_valid = [t0_dt + pd.Timedelta(seconds=int(j * dt)) for j in idx_raw]
        t_sec_valid    = idx_raw * dt

        df_out = pd.DataFrame({
            'datetime':          datetime_valid,
            't_sec':             t_sec_valid,
            'y1_measured':       y1[idx_pack],
            'y1_reconvolved':    y1h_col[idx_pack],
            'c_reconstructed':   c_col[idx_pack],
        })
        fname = f'deconv_output_exp{i}.csv'
        df_out.to_csv(fname, index=False)
        print(f'  Exp {i} -> {fname} ({len(idx_pack)} pts, dt={dt}s)')

    print('\n=== DONE ===')
    plt.show()


if __name__ == '__main__':
    main()