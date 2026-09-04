"""
Plotting diagnostics for LOPAP deconvolution.

Converted from MATLAB: plot_irf_diagnostics.m, plot_recon_diagnostics.m
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _lag_to_I(lag, I, I_end):
    """Map lag values to step-response values.

    Parameters
    ----------
    lag : array_like
        Lag indices (0-based time steps).
    I : array_like
        Cumulative step-response vector.
    I_end : float
        Terminal value of I (used for lags beyond the vector length).

    Returns
    -------
    x : np.ndarray
        Step-response values at each lag position.
    """
    lag = np.asarray(lag, dtype=int).ravel()
    x = np.zeros(lag.shape)
    pos = lag >= 0
    # Clamp positive lags to valid index range (0-based)
    idx = np.minimum(lag[pos], len(I) - 1)
    x[pos] = I[idx]
    x[lag > (len(I) - 1)] = I_end
    return x


def _fit_ab_wls(x, y, w):
    """Weighted least-squares fit: y = b + A * x.

    Parameters
    ----------
    x, y, w : array_like
        Predictor, response, and weight vectors (all 1-D, same length).

    Returns
    -------
    A : float
        Slope.
    b : float
        Intercept.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()

    S0 = np.sum(w)
    Sx = np.sum(w * x)
    Sxx = np.sum(w * x * x)
    Sy = np.sum(w * y)
    Sxy = np.sum(w * x * y)

    det_M = S0 * Sxx - Sx * Sx
    if abs(det_M) < 1e-12:
        b = Sy / max(S0, 1e-30)
        A = 0.0
    else:
        b = (Sy * Sxx - Sx * Sxy) / det_M
        A = (S0 * Sxy - Sx * Sy) / det_M
    return A, b


def _get_event_idx(ev, n):
    """Return training-window indices for an event (0-based).

    Parameters
    ----------
    ev : dict
        Event dict with optional keys 'keep_win', 'idx_train', 'win_inst1', 'idx'.
    n : int
        Length of the signal.

    Returns
    -------
    idx : np.ndarray
        0-based indices into the signal.
    """
    if 'keep_win' in ev and ev['keep_win'] is not None and not ev['keep_win']:
        return np.array([], dtype=int)

    if 'idx_train' in ev and ev['idx_train'] is not None and len(ev['idx_train']) > 0:
        idx = np.asarray(ev['idx_train'], dtype=int).ravel()
    elif ('win_inst1' in ev and ev['win_inst1'] is not None
          and len(ev['win_inst1']) == 2):
        idx = np.arange(ev['win_inst1'][0], ev['win_inst1'][1] + 1, dtype=int)
    elif 'idx' in ev and ev['idx'] is not None and len(ev['idx']) > 0:
        idx = np.asarray(ev['idx'], dtype=int).ravel()
    else:
        return np.array([], dtype=int)

    # Keep only valid 0-based indices
    idx = idx[(idx >= 0) & (idx < n)]
    return idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_irf_diagnostics(sd_star, exp_pack, detail, cfg):
    """Visualize the trained IRF and per-event step-response fits.

    Creates two figures:

    * **Figure 1 -- IRF Overview** (two side-by-side panels):
      (a) Impulse response h(t) with td and tmax marked.
      (b) Cumulative step response I(t) with alpha lines.

    * **Figure 2 -- Per-event step-response fits** (grid of subplots):
      Normalized measured data vs. fitted I(t), annotated with R^2.

    Parameters
    ----------
    sd_star : dict
        Trained IRF result with keys: theta (len-6 array), h_total, I,
        dt, alpha, t_window, tmax.
    exp_pack : list of dict
        Packed experiments; each has 'y1' (1-D array) and 'events'.
    detail : dict
        Training detail with 'events_selected' or 'events_step'
        (list of lists of event dicts), 'delta_exp' (array),
        'selection_enabled' (bool).
    cfg : dict
        Configuration; may contain nested 'train' dict with 'edge_sigma'.
    """

    theta = np.asarray(sd_star['theta'], dtype=float).ravel()
    h_total = np.asarray(sd_star['h_total'], dtype=float).ravel()
    I_total = np.asarray(sd_star['I'], dtype=float).ravel()
    dt = int(sd_star['dt'])
    alpha = float(sd_star['alpha'])
    tmax = int(sd_star['tmax'])

    td = int(round(theta[0]))  # detectable delay

    # ===== Figure 1: IRF overview =====
    t_ax = np.arange(len(h_total)) * dt

    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.2))
    fig1.canvas.manager.set_window_title('IRF Overview')

    # (a) Impulse response h(t)
    ax = axes1[0]
    ax.plot(t_ax, h_total, 'b-', linewidth=1.5)
    ax.axvline(td, color='r', linestyle='--', linewidth=1.3,
               label=f'$t_d$ = {td} s')
    ax.axvline(tmax, color='m', linestyle='--', linewidth=1.3,
               label=f'$t_{{max}}$ = {tmax} s')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('h(t)')
    theta_str = '  '.join(f'{v:.1f}' for v in theta)
    ax.set_title(f'Impulse Response Function\n$\\theta$ = [{theta_str}]')
    ax.set_xlim(0, min(t_ax[-1], tmax + 300))
    ax.legend(fontsize=8)
    ax.grid(True)

    # (b) Cumulative step response I(t)
    ax = axes1[1]
    ax.plot(t_ax, I_total, 'b-', linewidth=1.5)
    ax.axvline(td, color='r', linestyle='--', linewidth=1.3,
               label=f'$t_d$ = {td} s')
    ax.axvline(tmax, color='m', linestyle='--', linewidth=1.3,
               label=f'$t_{{max}}$ = {tmax} s')
    ax.axhline(alpha, color='r', linestyle=':', linewidth=1.0,
               label=f'$\\alpha$ = {alpha:.2f}')
    ax.axhline(1 - alpha, color='r', linestyle=':', linewidth=1.0,
               label=f'$1-\\alpha$ = {1 - alpha:.2f}')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('I(t)')
    ax.set_title(f'Step Response (CDF)\n$T_{{mem}}$ = {tmax - td} s')
    ax.set_xlim(0, min(t_ax[-1], tmax + 300))
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True)

    fig1.suptitle(
        f'Dual Inverse Gaussian IRF  |  '
        f'$\\tau_f$={theta[1]:.0f}  $Pe_f$={theta[2]:.1f}  '
        f'$w$={theta[3]:.2f}  $\\tau_s$={theta[4]:.0f}  '
        f'$Pe_s$={theta[5]:.1f}',
        fontsize=11,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.92])

    # ===== Figure 2: Per-event step response fits =====

    # Choose events (prefer selected if available)
    if (detail.get('selection_enabled', False)
            and 'events_selected' in detail
            and detail['events_selected'] is not None):
        events_cell = detail['events_selected']
        ev_label = 'selected'
    else:
        events_cell = detail.get('events_step', [])
        ev_label = 'all step'

    # Collect all events across experiments
    all_evs = []
    all_exp_id = []
    for i, evs in enumerate(events_cell):
        if evs is None or len(evs) == 0:
            continue
        for ev in evs:
            all_evs.append(ev)
            all_exp_id.append(i)

    n_ev = len(all_evs)
    if n_ev == 0:
        print('  [plot_irf_diagnostics] No events to plot.')
        return

    I_end = I_total[-1]

    # Edge weight parameters
    edge_sigma = 400
    if isinstance(cfg.get('train'), dict):
        edge_sigma = cfg['train'].get('edge_sigma', edge_sigma)
    w_center = theta[0]

    # Per-experiment delay delta
    use_delta = False
    delta_exp = np.zeros(len(exp_pack))
    if 'delta_exp' in detail and detail['delta_exp'] is not None:
        delta_exp = np.asarray(detail['delta_exp'], dtype=float).ravel()
        use_delta = True

    # Layout
    n_cols = min(5, n_ev)
    n_rows = int(np.ceil(n_ev / n_cols))
    fig_h = min(2.5 * n_rows, 9.0)
    fig_w = min(2.8 * n_cols, 15.0)

    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                squeeze=False)
    fig2.canvas.manager.set_window_title('Step Response Fits')

    r2_all = np.full(n_ev, np.nan)

    for ie in range(n_ev):
        ev = all_evs[ie]
        i_exp = all_exp_id[ie]

        pack = exp_pack[i_exp]
        y1 = np.asarray(pack['y1'], dtype=float).ravel()
        n = len(y1)

        t0 = int(round(ev['t0']))

        # Get training window (0-based)
        idx = _get_event_idx(ev, n)
        if len(idx) < 50:
            continue

        delta_i = 0
        if use_delta and i_exp < len(delta_exp):
            delta_i = int(round(delta_exp[i_exp]))

        lag = idx - t0 - delta_i
        y = y1[idx]

        # Fit A, b: y ~ b + A * I(lag)
        x = _lag_to_I(lag, I_total, I_end)
        w = np.exp(-0.5 * ((lag - w_center) / edge_sigma) ** 2)
        A, b = _fit_ab_wls(x, y, w)

        y_fit = b + A * x

        # Weighted R^2
        ww = w ** 2
        ss_res = np.sum(ww * (y - y_fit) ** 2)
        mu_w = np.sum(ww * y) / max(np.sum(ww), 1e-30)
        ss_tot = np.sum(ww * (y - mu_w) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        r2_all[ie] = r2

        # Normalize for display
        if abs(A) > 1e-30:
            y_norm = (y - b) / A
        else:
            y_norm = y - b

        # Plot
        row, col = divmod(ie, n_cols)
        ax = axes2[row, col]
        ax.plot(lag, y_norm, '.', color='0.6', markersize=3)

        lag_model = np.arange(lag.min(), lag.max() + 1)
        I_model = _lag_to_I(lag_model, I_total, I_end)
        ax.plot(lag_model, I_model, 'b-', linewidth=1.5)

        ax.axvline(0, color='k', linestyle=':', linewidth=0.8)

        ev_type = ev.get('type', 'rise')
        ax.set_title(f'E{i_exp}-{ie} [{ev_type}]\n$R^2$={r2:.3f}',
                      fontsize=9)

        # Background colour by R^2
        if r2 < 0.8:
            ax.set_facecolor((1.0, 0.95, 0.95))
        elif r2 > 0.95:
            ax.set_facecolor((0.95, 1.0, 0.95))

        ax.set_xlim(lag.min(), lag.max())
        ax.set_ylim(-0.1, 1.15)
        ax.grid(True)

        if row == n_rows - 1:
            ax.set_xlabel('lag (s)')
        if col == 0:
            ax.set_ylabel('Normalized response')

    # Hide unused subplots
    for ie in range(n_ev, n_rows * n_cols):
        row, col = divmod(ie, n_cols)
        axes2[row, col].set_visible(False)

    r2_valid = r2_all[np.isfinite(r2_all)]
    if len(r2_valid) > 0:
        fig2.suptitle(
            f'Step Response Fits ({ev_label} events, n={n_ev})  |  '
            f'median $R^2$ = {np.median(r2_valid):.3f}  '
            f'[{r2_valid.min():.3f}, {r2_valid.max():.3f}]',
            fontsize=11,
        )
    fig2.tight_layout(rect=[0, 0, 1, 0.93])

    print(f'  [IRF diagnostics] {n_ev} events plotted | '
          f'R^2: median={np.median(r2_valid):.3f}, '
          f'min={r2_valid.min():.3f}, max={r2_valid.max():.3f}')


def plot_recon_diagnostics(res, exp_pack, cfg):
    """Visualize deconvolution results for each experiment (valid region only).

    For each experiment creates a figure with two panels:

    * **Panel 1** -- Measured signal, reconvolved signal, and reconstructed
      concentration in the valid region (warm-up and padding excluded).
    * **Panel 2** -- Residual with RMSE annotation.

    Parameters
    ----------
    res : dict
        Deconvolution result with 'c_hat', 'y1_hat', and optionally
        'valid_lo', 'valid_hi' (each per experiment).
    exp_pack : list of dict
        Packed experiments; each has 'y1', 'n', and optionally 'prepad',
        'prepad_base', 'n_raw'.
    cfg : dict
        Configuration with 'dt_eff' (int), 'tmax_trained' (float).
    """

    n_exp = len(exp_pack)

    for i_exp in range(n_exp):
        pack = exp_pack[i_exp]
        c_hat = np.asarray(res['c_hat'][i_exp], dtype=float).ravel()
        y1_hat = np.asarray(res['y1_hat'][i_exp], dtype=float).ravel()

        n0 = int(pack['n'])

        dt_plot = max(1, int(cfg.get('dt_eff', 1)))

        y1 = np.asarray(pack['y1'], dtype=float).ravel()[:n0]
        y1h = y1_hat[:min(len(y1_hat), n0)]

        # ---- Valid region bounds ----
        prepad = int(pack.get('prepad', pack.get('prepad_base', 0)) or 0)
        n_raw = int(pack.get('n_raw', 0))
        tailpad_start = prepad + n_raw

        # Use pre-computed bounds from res if available; else compute
        if 'valid_lo' in res and len(res['valid_lo']) > i_exp:
            valid_lo = int(res['valid_lo'][i_exp])
            valid_hi = int(res['valid_hi'][i_exp])
        else:
            warmup_idx = prepad
            if cfg.get('tmax_trained', 0) > 0:
                warmup_idx = max(warmup_idx,
                                 int(round(cfg['tmax_trained'] / dt_plot)))
            valid_lo = warmup_idx
            valid_hi = tailpad_start if 0 < tailpad_start < n0 else n0

        # ---- Trim to valid region ----
        y1_v = y1[valid_lo:valid_hi]
        y1h_v = y1h[valid_lo:min(len(y1h), valid_hi)]
        nv = min(len(y1_v), len(y1h_v))
        y1_v = y1_v[:nv]
        y1h_v = y1h_v[:nv]

        c_hi = min(len(c_hat), valid_hi)
        c_v = c_hat[valid_lo:c_hi] if valid_lo < c_hi else np.array([])

        # Time axis: seconds from raw measurement start
        t_y = (np.arange(nv) + valid_lo - prepad) * dt_plot
        t_c = (np.arange(len(c_v)) + valid_lo - prepad) * dt_plot

        # ---- R² ----
        ss_res = np.sum((y1_v - y1h_v) ** 2)
        ss_tot = np.sum((y1_v - np.mean(y1_v)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        fig.canvas.manager.set_window_title(f'Exp{i_exp} Deconvolution')

        # ---- Panel 1: Time series (valid region only) ----
        ax = axes[0]
        ax.plot(t_y, y1_v, 'b-', linewidth=0.8, label='Measured')
        ax.plot(t_y, y1h_v, 'k--', linewidth=1.2, label='Reconvolved')
        if len(c_v) > 0:
            ax.plot(t_c, c_v, 'r-', linewidth=1.2, label='Reconstructed')
        ax.set_title(f'Exp {i_exp} | $R^2$ = {r2:.4f}')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlabel('t (s)')
        ax.set_ylabel('Concentration')
        ax.grid(True)

        # ---- Panel 2: Residual ----
        ax = axes[1]
        residual = y1_v - y1h_v
        rmse = np.sqrt(np.mean(residual ** 2))
        ax.plot(t_y, residual, 'r-', linewidth=0.5)
        ax.set_title(f'Residual | RMSE = {rmse:.4f}')
        ax.set_xlabel('t (s)')
        ax.set_ylabel('$y_1 - \\hat{y}_1$')
        ax.grid(True)

        fig.tight_layout()