"""
Nonnegative deconvolution with TV regularization (IRLS) and dual inverse-Gaussian
IRF builder for LOPAP data.

Public API
----------
- estimate_c_given_kernel : TV-regularised nonneg deconvolution via IRLS
- run_fixed_irf_recon     : batch deconvolution with a pre-trained dual-IG IRF
- irf_from_theta          : build a dual inverse-Gaussian IRF from 6 parameters
- rtd_ig                  : single inverse-Gaussian residence-time distribution
"""

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear


# ---------------------------------------------------------------------------
# RTD / IRF helpers
# ---------------------------------------------------------------------------

def rtd_ig(t, tau, Pe):
    """Inverse-Gaussian residence-time distribution.

    Parameters
    ----------
    t : array_like
        Time grid (s).  Values <= 0 yield h = 0.
    tau : float
        Mean residence time.
    Pe : float
        Peclet number.

    Returns
    -------
    h : ndarray
        Probability density at each *t*.
    """
    t = np.asarray(t, dtype=float)
    h = np.zeros_like(t)
    m = t > 0
    th = t[m] / tau
    h[m] = (1.0 / tau) * np.sqrt(Pe / (4.0 * np.pi * th ** 3)) * \
           np.exp(-Pe * (1.0 - th) ** 2 / (4.0 * th))
    return h


def irf_from_theta(theta, t_window, dt, alpha=0.05):
    """Build dual inverse-Gaussian IRF from a 6-element parameter vector.

    Parameters
    ----------
    theta : array_like, shape (6,)
        [delay_det, tau_f, Pe_f, w, tau_s, Pe_s].
    t_window : float
        Maximum time window (s).
    dt : float
        Time step (s).
    alpha : float, optional
        CDF fraction used to locate the onset of the fast component
        (default 0.05).

    Returns
    -------
    h_total : ndarray
        Normalised IRF kernel (sum * dt ~ 1).
    I : ndarray
        Cumulative distribution of *h_total*.
    diag : dict
        Diagnostic fields (alpha, lag_alpha, d_start, delay_det).
    """
    theta = np.asarray(theta, dtype=float).ravel()
    delay_det = max(0.0, theta[0])
    alpha = max(0.001, min(0.999, alpha))
    t = np.arange(0, t_window + dt * 0.5, dt)

    tau_f = max(1.0, theta[1]);  Pe_f = max(1e-3, theta[2])
    w     = max(0.01, min(0.99, theta[3]))
    tau_s = max(1.0, theta[4]);  Pe_s = max(1e-3, theta[5])

    # --- fast & slow components ---
    h_fast = rtd_ig(t, tau_f, Pe_f)
    h_fast = np.maximum(h_fast, 0.0)
    sf = np.sum(h_fast) * dt
    if sf > 0:
        h_fast /= sf

    h_slow = rtd_ig(t, tau_s, Pe_s)
    h_slow = np.maximum(h_slow, 0.0)
    ss = np.sum(h_slow) * dt
    if ss > 0:
        h_slow /= ss

    # --- mixture ---
    h_rtd = w * h_fast + (1.0 - w) * h_slow
    h_rtd /= (np.sum(h_rtd) * dt + np.finfo(float).eps)

    # --- onset via fast CDF ---
    h_onset = h_fast / (np.sum(h_fast) * dt + np.finfo(float).eps)
    I_onset = np.cumsum(h_onset) * dt
    indices = np.where(I_onset >= alpha)[0]
    k = indices[0] if len(indices) > 0 else 0
    lag_alpha = k  # already 0-based

    d_start = max(0, round(delay_det / dt) - lag_alpha)

    h_total = np.concatenate([np.zeros(d_start), h_rtd])
    h_total /= (np.sum(h_total) * dt + np.finfo(float).eps)
    I = np.cumsum(h_total) * dt
    I = np.clip(I, 0.0, 1.0)

    diag = dict(alpha=alpha, lag_alpha=lag_alpha,
                d_start=d_start, delay_det=delay_det)
    return h_total, I, diag


# ---------------------------------------------------------------------------
# Sparse convolution matrix
# ---------------------------------------------------------------------------

def _convmtx_causal_sparse(h, n0, n):
    """Build sparse causal convolution matrix H (n0 x n).

    ``y = H @ c`` performs causal convolution of *c* with kernel *h*,
    returning the first *n0* samples.

    Parameters
    ----------
    h : ndarray, shape (L,)
        Causal kernel starting at lag 0.
    n0 : int
        Number of output rows (observed signal length).
    n : int
        Number of columns (unknown signal length, >= n0).

    Returns
    -------
    H : sparse CSR matrix, shape (n0, n)
    """
    L = len(h)
    if L == 0:
        return sp.csr_matrix((n0, n))

    maxk = min(L - 1, n0 - 1)

    # Pre-count nonzeros
    nnz_total = 0
    for k in range(maxk + 1):
        r1 = k              # 0-based first row
        r2 = min(n0 - 1, n - 1 + k)  # 0-based last row
        if r2 >= r1:
            nnz_total += (r2 - r1 + 1)

    rows = np.empty(nnz_total, dtype=np.int32)
    cols = np.empty(nnz_total, dtype=np.int32)
    vals = np.empty(nnz_total, dtype=float)
    cnt = 0

    for k in range(maxk + 1):
        r1 = k
        r2 = min(n0 - 1, n - 1 + k)
        if r2 < r1:
            continue
        nm = r2 - r1 + 1
        rr = np.arange(r1, r2 + 1, dtype=np.int32)
        rows[cnt:cnt + nm] = rr
        cols[cnt:cnt + nm] = rr - k
        vals[cnt:cnt + nm] = h[k]
        cnt += nm

    H = sp.coo_matrix((vals, (rows, cols)), shape=(n0, n)).tocsr()
    return H


# ---------------------------------------------------------------------------
# Core deconvolution
# ---------------------------------------------------------------------------

def estimate_c_given_kernel(pack, h_in, reg, opt=None, x0=None):
    """Nonnegative deconvolution with TV regularisation (IRLS).

    Parameters
    ----------
    pack : dict
        Must contain ``'y1'`` (1-D signal). Optional ``'n'`` truncates *y1*.
    h_in : array_like
        Causal kernel (will be normalised to unit sum).
    reg : dict
        Regularisation parameters.  Recognised keys:

        - prepad (int) : number of pre-padded zeros in *y1* (default 0)
        - postpad (int) : extra unknown samples appended after *y1* (default 0)
        - prepad_anchor (int) : how many leading samples to clamp to 0
          (default = prepad)
        - lambda_tv (float) : TV penalty weight (default 0)
        - tv_eps (float) : Huber-like smoothing for TV (default 1e-3)
        - tv_maxit (int) : max IRLS iterations (default 8)
        - lambda_end0 (float) : tail-clamp penalty weight (default 0)
        - tailpad (int) : how many trailing samples are penalised (default 0)
        - downsample_factor (int) : manual downsample factor (default 1)
        - max_points (int) : auto-downsample threshold (default 30000)
    opt : dict or None, optional
        Solver options.  Recognised key: ``'tol'`` passed to
        `scipy.optimize.lsq_linear` as *tol*.
    x0 : array_like or None, optional
        Initial guess for *c*.

    Returns
    -------
    c_hat : ndarray
        Reconstructed concentration.
    y1_hat : ndarray
        Reconvolved signal (H @ c_hat).
    beta_hat : None
        Legacy placeholder.
    x_hat : ndarray
        Same as *c_hat*.
    """
    reg = dict(reg)  # local copy so we can mutate

    y1 = np.asarray(pack['y1'], dtype=float).ravel()
    n0 = pack.get('n') or len(y1)
    y1 = y1[:n0].copy()

    h = np.asarray(h_in, dtype=float).ravel()
    if len(h) == 0 or not np.all(np.isfinite(h)):
        raise ValueError('estimate_c_given_kernel: invalid kernel h_in')

    # ---- downsampling bookkeeping ----
    ds = 1
    n0_orig = n0
    y1_orig = y1.copy()
    postpad_orig = 0

    ds_manual = reg.get('downsample_factor', 1)
    if ds_manual is not None and ds_manual > 1:
        ds = int(round(ds_manual))

    max_pts = reg.get('max_points', 30000)
    if max_pts and max_pts > 0 and n0 > max_pts and ds == 1:
        ds = int(np.ceil(n0 / max_pts))
        print(f'  [Memory protection] n={n0} > {max_pts}, auto downsample by {ds}')

    if ds > 1:
        y1 = y1[::ds]
        n0 = len(y1)

        # average-pool kernel to preserve area
        h_len = len(h)
        n_ds = int(np.ceil(h_len / ds))
        h_ds = np.zeros(n_ds)
        for i in range(n_ds):
            idx_start = i * ds
            idx_end = min((i + 1) * ds, h_len)
            h_ds[i] = np.sum(h[idx_start:idx_end])
        h = h_ds

        postpad_orig = reg.get('postpad', 0)
        if 'prepad' in reg:
            reg['prepad'] = round(reg['prepad'] / ds)
        if 'tailpad' in reg:
            reg['tailpad'] = round(reg['tailpad'] / ds)
        if 'postpad' in reg:
            reg['postpad'] = round(reg['postpad'] / ds)

    # ---- defaults ----
    reg.setdefault('prepad', 0)
    reg.setdefault('postpad', 0)
    reg.setdefault('lambda_tv', 0)
    reg.setdefault('tv_eps', 1e-3)
    reg.setdefault('tv_maxit', 8)
    reg.setdefault('lambda_end0', 0)
    reg.setdefault('tailpad', 0)

    P = max(0, min(round(reg['prepad']), n0))
    Lp = max(0, round(reg['postpad']))
    n = n0 + Lp

    # prepad_anchor: only clamp this many leading samples to 0
    P0 = P
    pa = reg.get('prepad_anchor')
    if pa is not None and np.isfinite(pa):
        P0 = max(0, min(round(pa), P))

    # normalise kernel
    h = np.maximum(h, 0.0)
    h = h / (np.sum(h) + np.finfo(float).eps)

    H = _convmtx_causal_sparse(h, n0, n)

    # first-difference operator D1: (n-1) x n
    D1 = sp.diags([-np.ones(n), np.ones(n)], [0, 1],
                  shape=(n - 1, n), format='csr')

    # ---- bounds ----
    lb = np.zeros(n)
    ub = np.full(n, np.inf)

    # near-equality constraints via bounds: clamp c[0:P0] ~ 0
    # (lsq_linear requires lb < ub strictly, so use a tiny upper bound)
    if P0 > 0:
        lb[:P0] = 0.0
        ub[:P0] = 1e-30

    # ---- initial guess ----
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
        if len(x0) >= n and np.all(np.isfinite(x0[:n])):
            c = x0[:n].copy()
        else:
            c = np.maximum(H.T @ y1, 0.0)
            if len(c) < n:
                c = np.concatenate([c, np.zeros(n - len(c))])
    else:
        c = np.maximum(H.T @ y1, 0.0)
        if len(c) < n:
            c = np.concatenate([c, np.zeros(n - len(c))])
    if P0 > 0:
        c[:P0] = 0.0

    lambda_tv = reg['lambda_tv']

    # ---- tail soft-clamp ----
    tp = max(0, round(reg['tailpad']))
    if reg['lambda_end0'] > 0 and tp > 0:
        idx_start = max(0, n - tp)
        idx = np.arange(idx_start, n)
        n_idx = len(idx)
        It = sp.coo_matrix(
            (np.ones(n_idx), (np.arange(n_idx), idx)),
            shape=(n_idx, n)
        ).tocsr()
        Atail = np.sqrt(reg['lambda_end0']) * It
        btail = np.zeros(n_idx)
    else:
        Atail = sp.csr_matrix((0, n))
        btail = np.zeros(0)

    # ---- solver tolerance ----
    solver_tol = 1e-10
    if opt is not None and isinstance(opt, dict):
        solver_tol = opt.get('tol', 1e-10)

    # ---- IRLS loop for TV ----
    max_it = max(1, reg['tv_maxit'])
    for it in range(max_it):
        if lambda_tv > 0:
            d = D1 @ c
            w = 1.0 / np.sqrt(d ** 2 + reg['tv_eps'] ** 2)
            W = sp.diags(np.sqrt(w), 0, shape=(n - 1, n - 1), format='csr')
            Atv = np.sqrt(lambda_tv) * (W @ D1)
            btv = np.zeros(n - 1)
            A = sp.vstack([H, Atv, Atail], format='csr')
            b = np.concatenate([y1, btv, btail])
        else:
            A = sp.vstack([H, Atail], format='csr')
            b = np.concatenate([y1, btail])

        result = lsq_linear(A, b, bounds=(lb, ub), tol=solver_tol,
                            max_iter=500, method='trf')
        c_new = result.x

        rel_change = np.linalg.norm(c_new - c) / (np.linalg.norm(c) + np.finfo(float).eps)
        if rel_change < 1e-3:
            c = c_new
            break
        c = c_new

    c_hat = c
    y1_hat = (H @ c_hat)

    # ---- upsample back if downsampled ----
    if ds > 1:
        c_hat_ds = c_hat
        n_orig = n0_orig + max(0, postpad_orig)
        x_ds = np.arange(len(c_hat_ds))
        x_orig = np.linspace(0, len(c_hat_ds) - 1, n_orig)
        c_hat = np.interp(x_orig, x_ds, c_hat_ds)
        c_hat = np.maximum(c_hat, 0.0)

        # recompute y1_hat with original-resolution kernel
        h_orig = np.asarray(h_in, dtype=float).ravel()
        h_orig = np.maximum(h_orig, 0.0)
        h_orig = h_orig / (np.sum(h_orig) + np.finfo(float).eps)
        y1_hat = np.convolve(c_hat, h_orig, 'full')[:n0_orig]

        print(f'  [Upsample] c_hat: {len(c_hat_ds)} -> {len(c_hat)}, '
              f'y1_hat: {len(y1_hat)}')

    beta_hat = None
    x_hat = c_hat
    return c_hat, y1_hat, beta_hat, x_hat


# ---------------------------------------------------------------------------
# Batch reconstruction
# ---------------------------------------------------------------------------

def run_fixed_irf_recon(exp_pack, sd_star, cfg):
    """Deconvolve LOPAP data using a pre-trained dual-IG IRF.

    Parameters
    ----------
    exp_pack : list of dict
        Packed experiments (from ``pack_lopap_only``, deconv mode).
    sd_star : dict
        Trained IRF parameters.  Must contain ``'theta'`` (6-element array)
        and optionally ``'alpha'``.
    cfg : dict
        Configuration.  Expected keys:

        - dt_eff (float) : effective time step (default 1)
        - t_window (float) : kernel time window (s)
        - reg (dict) : regularisation parameters for ``estimate_c_given_kernel``
        - opt (dict) : solver options, e.g. ``{'lsqlin': {'tol': 1e-10}}``
        - quiet (bool) : suppress printing (default False)

    Returns
    -------
    res : dict
        Fields: ``c_hat``, ``y1_hat``, ``h_used``, ``tau_used``,
        ``r2_inst1``, ``dt_peak``, ``diag``, ``sd_star``, ``alpha``.
    """
    n_exp = len(exp_pack)
    quiet = cfg.get('quiet', False)

    dt_eff = cfg.get('dt_eff', 1)
    if dt_eff is None:
        dt_eff = 1

    # ---- theta ----
    theta_train = np.asarray(sd_star['theta'], dtype=float).ravel()
    shape_params = theta_train[1:6]  # tau_f, Pe_f, w, tau_s, Pe_s

    # ---- t_window ----
    twin = cfg.get('t_window')
    if twin is None:
        kernel_cfg = cfg.get('kernel', {})
        twin = kernel_cfg.get('t_window') if isinstance(kernel_cfg, dict) else None
    if twin is None:
        raise ValueError('run_fixed_irf_recon: cannot find t_window in cfg.')
    twin = round(twin)

    # ---- alpha ----
    alpha = sd_star.get('alpha')
    if alpha is None:
        alpha = cfg.get('alpha_onset', 0.05)

    if not quiet:
        print(f'  [Kernel] DUAL-IG: tau_f={shape_params[0]:.0f}, '
              f'Pe_f={shape_params[1]:.1f}, w={shape_params[2]:.3f}, '
              f'tau_s={shape_params[3]:.0f}, Pe_s={shape_params[4]:.1f}')
        print(f'  [dt_eff] = {dt_eff} s, alpha = {alpha:.3f}')

    # ---- init results ----
    res = dict(
        sd_star=sd_star,
        alpha=alpha,
        c_hat=[None] * n_exp,
        y1_hat=[None] * n_exp,
        h_used=[None] * n_exp,
        tau_used=np.full(n_exp, np.nan),
        r2_inst1=np.full(n_exp, np.nan),
        dt_peak=np.full(n_exp, np.nan),
        valid_lo=np.zeros(n_exp, dtype=int),
        valid_hi=np.zeros(n_exp, dtype=int),
        diag=[None] * n_exp,
    )

    t_all = time.time()

    for i in range(n_exp):
        pack = exp_pack[i]
        t0 = time.time()

        # ---- tau for this experiment ----
        tau_i = pack.get('tau')
        if tau_i is None:
            tau_i = theta_train[0]

        # ---- generate IRF on dt_eff grid ----
        tau_i_sec = tau_i * dt_eff
        theta_i = np.array([tau_i_sec, *shape_params])
        h_i, _, irf_diag = irf_from_theta(theta_i, twin, dt_eff, alpha)

        res['h_used'][i] = h_i
        res['tau_used'][i] = tau_i

        # ---- TV epsilon from signal range ----
        reg_i = dict(cfg['reg'])
        tv_eps_frac = cfg['reg'].get('tv_eps_frac', 0)
        if tv_eps_frac and tv_eps_frac > 0:
            prepad = pack.get('prepad', 0)
            n_raw = pack.get('n_raw')
            y1_full = np.asarray(pack['y1'], dtype=float).ravel()
            if prepad is not None and n_raw is not None:
                sig_i = y1_full[prepad:min(prepad + n_raw, len(y1_full))]
            else:
                sig_i = y1_full
            rng_i = np.percentile(sig_i, 99) - np.percentile(sig_i, 1)
            if not np.isfinite(rng_i) or rng_i <= 0:
                rng_i = max(np.std(sig_i), np.finfo(float).eps)
            reg_i['tv_eps'] = tv_eps_frac * rng_i

        # ---- solver options ----
        opt_lsqlin = None
        opt_cfg = cfg.get('opt')
        if isinstance(opt_cfg, dict):
            opt_lsqlin = opt_cfg.get('lsqlin')

        # ---- deconvolve ----
        c_hat, y1_hat, _, _ = estimate_c_given_kernel(
            pack, h_i, reg_i, opt=opt_lsqlin, x0=None
        )

        # ---- trim tailpad ----
        prepad_base = pack.get('prepad_base', 0)
        n_raw = pack.get('n_raw', len(c_hat))
        n_valid_c = prepad_base + n_raw
        if len(c_hat) > n_valid_c:
            c_hat = c_hat[:n_valid_c]

        res['c_hat'][i] = c_hat
        res['y1_hat'][i] = y1_hat

        # ---- R-squared (valid region only: skip warmup/padding) ----
        y1 = np.asarray(pack['y1'], dtype=float).ravel()
        n0 = pack.get('n', len(y1))
        y1 = y1[:n0]
        y1h = y1_hat[:min(len(y1_hat), n0)]
        if len(y1h) < n0:
            y1h = np.concatenate([y1h, np.zeros(n0 - len(y1h))])

        # valid range: after warmup, before tailpad
        prepad_v = pack.get('prepad', pack.get('prepad_base', 0))
        n_raw_v = pack.get('n_raw', n0)
        kbase_deconv = cfg.get('Kbase', 0) if cfg is not None else 0
        tmax_trained = cfg.get('tmax_trained', 0) if cfg is not None else 0
        if kbase_deconv == 0 and tmax_trained > 0:
            warmup_v = round(tmax_trained / dt_eff)
        else:
            warmup_v = prepad_v
        idx_lo_v = max(warmup_v, prepad_v)
        idx_hi_v = prepad_v + n_raw_v
        if idx_hi_v <= 0 or idx_hi_v > n0:
            idx_hi_v = n0

        y1_valid = y1[idx_lo_v:idx_hi_v]
        y1h_valid = y1h[idx_lo_v:idx_hi_v]
        SS_tot = np.sum((y1_valid - np.mean(y1_valid)) ** 2)
        SS_res = np.sum((y1_valid - y1h_valid) ** 2)
        res['r2_inst1'][i] = 1.0 - SS_res / max(SS_tot, np.finfo(float).eps)

        res['valid_lo'][i] = idx_lo_v
        res['valid_hi'][i] = idx_hi_v

        res['dt_peak'][i] = np.nan
        res['diag'][i] = dict(note='ok')

        if not quiet:
            elapsed = time.time() - t0
            print(f'  Exp {i + 1}/{n_exp} | tau={tau_i:.0f} | '
                  f'R2={res["r2_inst1"][i]:.4f} | {elapsed:.1f}s')

    if not quiet:
        total = time.time() - t_all
        print(f'>>> Deconvolution finished. total elapsed={total:.1f}s')

    return res