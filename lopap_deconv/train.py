"""
IRF training for LOPAP deconvolution.

Estimates the dual inverse Gaussian IRF from step-response data
using Latin Hypercube search + bounded optimisation (L-BFGS-B / SLSQP).
"""

import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from .deconvolve import irf_from_theta, rtd_ig


# ========================================================================
#  Helper: get series from pack
# ========================================================================

def _get_series(pack, f1, f2):
    if f1 in pack and pack[f1] is not None:
        return np.asarray(pack[f1], dtype=float).ravel()
    if f2 in pack and pack[f2] is not None:
        return np.asarray(pack[f2], dtype=float).ravel()
    raise KeyError(f"Missing field '{f1}' or '{f2}' in pack.")


# ========================================================================
#  Helper: weighted least-squares  y = b + A*x
# ========================================================================

def _fit_ab_wls(x, y, w):
    x, y, w = (np.asarray(v, dtype=float).ravel() for v in (x, y, w))
    S0 = np.sum(w)
    Sx = np.sum(w * x)
    Sxx = np.sum(w * x * x)
    Sy = np.sum(w * y)
    Sxy = np.sum(w * x * y)
    det_M = S0 * Sxx - Sx * Sx
    eps = 1e-12
    if abs(det_M) < eps:
        return 0.0, Sy / max(S0, eps)
    A = (S0 * Sxy - Sx * Sy) / det_M
    b = (Sy * Sxx - Sx * Sxy) / det_M
    return A, b


def _lag_to_I(lag, I, I_end):
    lag = np.asarray(lag, dtype=int).ravel()
    x = np.zeros_like(lag, dtype=float)
    pos = lag >= 0
    idx = np.minimum(lag[pos], len(I) - 1)
    x[pos] = I[idx]
    x[lag > len(I) - 1] = I_end
    return x


def _get_event_train_idx(ev, n):
    """Get training window indices (0-based) for an event."""
    if ev.get('keep_win') is not None and not ev['keep_win']:
        return np.array([], dtype=int)
    if 'idx_train' in ev and ev['idx_train'] is not None and len(ev['idx_train']) > 0:
        idx = np.asarray(ev['idx_train'], dtype=int).ravel()
    elif 'win_inst1' in ev and ev['win_inst1'] is not None and len(ev['win_inst1']) == 2:
        idx = np.arange(ev['win_inst1'][0], ev['win_inst1'][1] + 1, dtype=int)
    elif 'idx' in ev and ev['idx'] is not None and len(ev['idx']) > 0:
        idx = np.asarray(ev['idx'], dtype=int).ravel()
    else:
        return np.array([], dtype=int)
    return idx[(idx >= 0) & (idx < n)]


# ========================================================================
#  tmin/tmax estimation from events
# ========================================================================

def _find_inst1_onset(dy1, t0, direction, p):
    """Find onset in derivative signal after t0."""
    n = len(dy1)
    sgn = 1 if direction == 'rise' else -1

    b0 = max(0, t0 - 1200)
    b1 = max(0, t0 - 200)
    if b1 <= b0 + 50:
        b0 = max(0, t0 - 800)
        b1 = max(0, t0 - 100)

    base_dy = dy1[b0:b1 + 1]
    sig = 1.4826 * np.median(np.abs(base_dy - np.median(base_dy))) + np.finfo(float).eps
    thr = p.get('onset_k', 1.5) * sig

    s0 = min(n - 1, t0 + 1)
    s1 = min(n - 1, t0 + int(round(p.get('search_max', 2500))))
    persist = int(round(p.get('onset_persist', 20)))
    if s1 <= s0 + persist:
        return np.nan

    z = sgn * dy1[s0:s1 + 1]
    run = 0
    for k in range(len(z)):
        if z[k] > thr:
            run += 1
            if run >= persist:
                return s0 + k - persist + 1
        else:
            run = 0
    return np.nan


def _find_inst1_end_by_derivative(dy1, t_start, p, t_bl):
    """Find end of response (derivative returns to baseline noise)."""
    n = len(dy1)

    b0 = max(0, t_bl - 1200)
    b1 = max(0, t_bl - 200)
    if b1 <= b0 + 50:
        b0 = max(0, t_bl - 800)
        b1 = max(0, t_bl - 100)
    if b1 <= b0:
        return np.nan

    base_dy = dy1[b0:b1 + 1]
    sig = 1.4826 * np.median(np.abs(base_dy - np.median(base_dy))) + np.finfo(float).eps
    thr_end = p.get('end_k', 2.0) * sig
    L = max(5, int(round(p.get('end_hold', 120))))

    s0 = min(n - 1, int(round(t_start)) + 1)
    s1 = min(n - 1, int(round(t_start)) + int(round(p.get('search_max', 2500))))
    if s1 <= s0 + L:
        return np.nan

    z = np.abs(dy1[s0:s1 + 1])
    run = 0
    for k in range(len(z)):
        if z[k] <= thr_end:
            run += 1
            if run >= L:
                return s0 + k - L + 1
        else:
            run = 0
    return np.nan


def _estimate_tmin_tmax_from_events(y1, events_step, cfg_tm):
    """Estimate tmin and tmax from step events."""
    from scipy.ndimage import uniform_filter1d

    y1 = np.asarray(y1, dtype=float).ravel()
    n = len(y1)

    p = {
        'y1_smooth_win': 31, 'search_max': 2500,
        'onset_k': 1.5, 'onset_persist': 20,
        'end_k': 2.0, 'end_hold': 120,
        'min_delay_for_tmax': 400, 'tmax_cap': 2500,
        'use_quantile_tmax': 0.8,
    }
    if cfg_tm is not None:
        p.update({k: v for k, v in cfg_tm.items() if v is not None})

    sw = max(5, int(round(p['y1_smooth_win'])))
    y1s = uniform_filter1d(y1, size=sw, mode='nearest')
    dy1 = np.diff(y1s, append=0)

    tmin_list = []
    tmax_list = []

    for ev in events_step:
        t0 = ev.get('t0')
        if t0 is None:
            continue
        t0 = int(round(t0))
        if t0 < 0 or t0 >= n:
            continue

        direction = ev.get('type', 'rise')
        is_fall = (direction == 'fall')

        ton = _find_inst1_onset(dy1, t0, direction, p)
        if np.isfinite(ton):
            tmin_list.append(ton - t0)

        if not is_fall:
            continue

        t_ss = ton if np.isfinite(ton) else min(n - 1, t0 + int(round(p['min_delay_for_tmax'])))

        tend = _find_inst1_end_by_derivative(dy1, t_ss, p, t0)
        if np.isfinite(tend):
            tmax_i = tend - t0
            if tmax_i <= p['tmax_cap']:
                tmax_list.append(tmax_i)

    tmin_hat = int(round(np.median(tmin_list))) if tmin_list else np.nan
    if tmax_list:
        tmax_hat = int(round(np.quantile(tmax_list, p['use_quantile_tmax'])))
    else:
        tmax_hat = np.nan

    return tmin_hat, tmax_hat


# ========================================================================
#  Event windowing and selection
# ========================================================================

def _add_event_windows(events, n, tmin_hat, tmax_hat, cfg_win):
    """Add training windows to events based on tmin/tmax."""
    if not events:
        return events

    guard_pre = cfg_win.get('guard_pre', 250)
    guard_post = cfg_win.get('guard_post', cfg_win.get('guard', 60))
    min_len = cfg_win.get('min_len', 200)
    disable_mid_clip = cfg_win.get('disable_mid_clip', False)
    pre_need = int(round(cfg_win.get('pre_need', 200)))

    tmin = tmin_hat if np.isfinite(tmin_hat) else 500
    tmax = tmax_hat if np.isfinite(tmax_hat) else 1000

    # Add t_eff and sort
    for ev in events:
        ev['t_eff'] = ev['t0'] + tmin
    events.sort(key=lambda e: e['t_eff'])

    te = np.array([e['t_eff'] for e in events], dtype=float)
    mid = (te[:-1] + te[1:]) / 2.0

    for i, ev in enumerate(events):
        t0 = int(round(ev['t0']))
        lo = int(round(t0 + tmin - guard_pre))
        hi = int(round(t0 + tmax + guard_post))

        if not disable_mid_clip:
            if i > 0:
                lo = max(lo, int(np.floor(mid[i - 1])) + 1)
            if i < len(events) - 1:
                hi = min(hi, int(np.ceil(mid[i])) - 1)

        lo = max(0, lo)
        hi = min(n - 1, hi)

        ev['win_inst1'] = [lo, hi]
        ev['win_len'] = hi - lo + 1

        keep = ev['win_len'] >= min_len
        if keep and (lo - t0) > (tmin - pre_need):
            keep = False
        ev['keep_win'] = keep
        ev['idx_train'] = np.arange(lo, hi + 1, dtype=int) if keep else np.array([], dtype=int)

    return events


def _select_isolated_events(events, greedy=True):
    """Select non-overlapping events greedily."""
    if not events:
        return events

    kept = [e for e in events if e.get('keep_win', False)]
    if not kept:
        return kept

    kept.sort(key=lambda e: e['win_inst1'][0])

    if greedy:
        chosen = []
        last_hi = -np.inf
        for ev in kept:
            if ev['win_inst1'][0] > last_hi:
                chosen.append(ev)
                last_hi = ev['win_inst1'][1]
        return chosen
    return kept


# ========================================================================
#  Objective function
# ========================================================================

def _kernel_objective(phi, delay_fixed, exp_pack, t_window, train, events_cell, dt):
    """Compute mean SSE across events for given IRF shape parameters.

    Parameters
    ----------
    phi : array_like, shape (5,)
        [tau_f, Pe_f, w, tau_s, Pe_s]
    delay_fixed : float
        Fixed detectable delay.
    """
    theta = np.array([delay_fixed, *phi])
    alpha = train.get('anchor_alpha', 0.05)

    h_vec, I, _ = irf_from_theta(theta, t_window, dt, alpha)
    I_end = I[-1]

    w_center = delay_fixed
    if 'w_center_fixed' in train and np.isfinite(train['w_center_fixed']):
        w_center = train['w_center_fixed']
    edge_sigma = train.get('edge_sigma', 400)

    use_delta = train.get('use_per_exp_delta', False)
    delta_exp = train.get('delta_exp', np.array([]))

    use_shift = train.get('use_micro_shift', False)
    S = max(0, int(round(train.get('shift_max', 0)))) if use_shift else 0
    lambda_shift = train.get('lambda_shift', 0) if use_shift else 0

    total = 0.0
    cnt = 0

    for i, pack in enumerate(exp_pack):
        y1 = _get_series(pack, 'y1', 'inst1')
        n = len(y1)

        delta_i = 0
        if use_delta and len(delta_exp) > i and np.isfinite(delta_exp[i]):
            delta_i = int(round(delta_exp[i]))

        evs = events_cell[i] if i < len(events_cell) else []
        for ev in evs:
            t0 = ev.get('t0')
            if t0 is None:
                continue
            t0 = int(round(t0))
            idx = _get_event_train_idx(ev, n)
            if len(idx) < 50:
                continue

            lag_adj = idx - t0 - delta_i
            y = y1[idx]

            if not use_shift or S == 0:
                x = _lag_to_I(lag_adj, I, I_end)
                w = np.exp(-0.5 * ((lag_adj - w_center) / edge_sigma) ** 2)
                A, b = _fit_ab_wls(x, y, w)
                r = w * (y - (b + A * x))
                total += np.sum(r ** 2)
            else:
                best = np.inf
                for dt0 in range(-S, S + 1):
                    lag_phys = lag_adj - dt0
                    x = _lag_to_I(lag_phys, I, I_end)
                    w = np.exp(-0.5 * ((lag_phys - w_center) / edge_sigma) ** 2)
                    A, b = _fit_ab_wls(x, y, w)
                    r = w * (y - (b + A * x))
                    sse = np.sum(r ** 2) + lambda_shift * dt0 ** 2
                    if sse < best:
                        best = sse
                total += best
            cnt += 1

    return total / cnt if cnt > 0 else 1e30


# ========================================================================
#  Event scoring and selection
# ========================================================================

def _score_events_by_fit(exp_pack, events_cell, theta_ref, t_window, train, dt):
    """Score each event by its fit to the reference IRF."""
    alpha = train.get('anchor_alpha', 0.05)
    _, I, _ = irf_from_theta(theta_ref, t_window, dt, alpha)
    I_end = I[-1]

    w_center = theta_ref[0]
    if 'w_center_fixed' in train and np.isfinite(train['w_center_fixed']):
        w_center = train['w_center_fixed']
    edge_sigma = train.get('edge_sigma', 120)

    use_delta = train.get('use_per_exp_delta', False)
    delta_exp = train.get('delta_exp', np.array([]))

    use_shift = train.get('use_micro_shift', False)
    S = max(0, int(round(train.get('shift_max', 0)))) if use_shift else 0
    lambda_shift = train.get('lambda_shift', 0) if use_shift else 0

    events_scored = []
    for i, pack in enumerate(exp_pack):
        y1 = _get_series(pack, 'y1', 'inst1')
        n = len(y1)

        delta_i = 0
        if use_delta and len(delta_exp) > i and np.isfinite(delta_exp[i]):
            delta_i = int(round(delta_exp[i]))

        evs = events_cell[i] if i < len(events_cell) else []
        evs = [dict(e) for e in evs]  # copy

        for ev in evs:
            ev.setdefault('fit_sse', np.inf)
            ev.setdefault('fit_r2', -np.inf)
            ev.setdefault('fit_sse_norm', np.inf)

            t0 = ev.get('t0')
            if t0 is None:
                continue
            t0 = int(round(t0))
            idx = _get_event_train_idx(ev, n)
            if len(idx) < 50:
                continue

            lag_adj = idx - t0 - delta_i
            y = y1[idx]

            best = np.inf
            best_sse = np.inf
            best_sst = np.inf

            dt0_list = [0] if (not use_shift or S == 0) else range(-S, S + 1)

            for dt0 in dt0_list:
                lag_phys = lag_adj - dt0
                x = _lag_to_I(lag_phys, I, I_end)
                w = np.exp(-0.5 * ((lag_phys - w_center) / edge_sigma) ** 2)
                A, b = _fit_ab_wls(x, y, w)
                yfit = b + A * x
                ww = w ** 2
                sse = np.sum(ww * (y - yfit) ** 2)
                muw = np.sum(ww * y) / max(np.sum(ww), np.finfo(float).eps)
                sst = np.sum(ww * (y - muw) ** 2)
                score = sse / max(sst, np.finfo(float).eps) + lambda_shift * dt0 ** 2
                if score < best:
                    best = score
                    best_sse = sse
                    best_sst = sst

            ev['fit_sse'] = best_sse
            ev['fit_r2'] = 1 - best_sse / max(best_sst, np.finfo(float).eps)
            ev['fit_sse_norm'] = best_sse / max(best_sst, np.finfo(float).eps)

        events_scored.append(evs)
    return events_scored


def _select_topk_events(events_scored_cell, train):
    """Select top-K events per type (rise/fall) per experiment."""
    K_rise = max(0, int(round(train.get('topk_rise', 3))))
    K_fall = max(0, int(round(train.get('topk_fall', 3))))
    min_sep = max(1, int(round(train.get('select_min_sep', 300))))

    result = []
    for evs in events_scored_cell:
        if not evs:
            result.append(evs)
            continue

        # Filter valid
        evs = [e for e in evs if e.get('type') in ('rise', 'fall')]
        evs = [e for e in evs if e.get('keep_win', True)]
        if not evs:
            result.append([])
            continue

        scores = np.array([e.get('fit_sse_norm', np.inf) for e in evs])

        chosen = set()

        def pick_topk(mask, K):
            if K <= 0:
                return
            idxs = [i for i in range(len(evs)) if mask[i]]
            if not idxs:
                return
            idxs.sort(key=lambda j: scores[j])
            picked = 0
            for j in idxs:
                if j in chosen:
                    continue
                t0_j = evs[j]['t0']
                too_close = any(abs(evs[c]['t0'] - t0_j) < min_sep for c in chosen)
                if too_close:
                    continue
                chosen.add(j)
                picked += 1
                if picked >= K:
                    break

        is_rise = [e['type'] == 'rise' for e in evs]
        is_fall = [e['type'] == 'fall' for e in evs]
        pick_topk(is_rise, K_rise)
        pick_topk(is_fall, K_fall)

        if train.get('select_keep_at_least_one_per_exp', True) and not chosen:
            chosen.add(int(np.argmin(scores)))

        ev_sel = [evs[j] for j in sorted(chosen)]
        ev_sel.sort(key=lambda e: e['t0'])
        result.append(ev_sel)

    return result


# ========================================================================
#  Main training function
# ========================================================================

def fit_irf_sd_train(exp_pack, cfg):
    """Estimate the dual inverse Gaussian IRF from step-response data.

    Parameters
    ----------
    exp_pack : list of dict
        Packed training experiments (from ``pack_lopap_only`` in train mode).
    cfg : dict
        Configuration dict (from ``cfg_default`` + overrides).
        Must contain ``cfg['train']['tau_known']``.

    Returns
    -------
    sd_star : dict
        Trained IRF: theta, h_total, I, tmin, tmax, alpha, etc.
    fval : float
        Final objective value.
    detail : dict
        Events, diagnostics, selection info.
    """
    dt = 1  # pipeline uses 1 s grid

    # ---- t_window ----
    twin = cfg.get('t_window')
    if twin is None:
        kernel_cfg = cfg.get('kernel', {})
        twin = kernel_cfg.get('t_window') if isinstance(kernel_cfg, dict) else None
    if twin is None:
        raise ValueError('fit_irf_sd_train: cannot find t_window in cfg.')
    twin = int(round(twin))

    # ---- training config ----
    train_defaults = {
        'W': 800, 'edge_sigma': 120,
        'max_events': 20, 'ev_min_sep': 300,
        'lopap_only': True, 'tau_known': None,
        'use_micro_shift': True, 'shift_max': 20, 'lambda_shift': 1e-3,
        'use_per_exp_delta': True,
        'do_select_topk': True, 'two_pass': True,
        'topk_rise': 3, 'topk_fall': 3,
        'select_min_sep': None,
        'select_keep_at_least_one_per_exp': True,
        'anchor_alpha': cfg.get('alpha_onset', 0.05),
        'auto_search': False,
        'search_N_coarse': 500, 'search_N_topk': 15,
        'search_auto_tighten': False,
        'use_isolated_events': True,
    }
    # timing sub-config
    train_defaults['tm'] = {
        'y1_smooth_win': 31, 'search_max': 2500,
        'onset_k': 1.5, 'onset_persist': 8,
        'end_k': 2.0, 'end_hold': 120,
        'min_delay_for_tmax': 400, 'tmax_cap': 2500,
        'use_quantile_tmax': 0.8,
    }
    # window sub-config
    train_defaults['win'] = {
        'guard_pre': 250, 'guard_post': 60,
        'guard': 60, 'min_len': 200,
        'disable_mid_clip': False, 'pre_need': 200,
    }

    train = dict(train_defaults)
    if 'train' in cfg and isinstance(cfg['train'], dict):
        for k, v in cfg['train'].items():
            if isinstance(v, dict) and k in train and isinstance(train[k], dict):
                train[k].update(v)
            else:
                train[k] = v
    # merge reg into train for backward compat
    if 'reg' in cfg and isinstance(cfg['reg'], dict):
        for k, v in cfg['reg'].items():
            if k not in train:
                train[k] = v
    if train['select_min_sep'] is None:
        train['select_min_sep'] = train['ev_min_sep']

    n_exp = len(exp_pack)
    if n_exp == 0:
        raise ValueError('fit_irf_sd_train: exp_pack is empty.')

    # ---- LOPAP-only: tau from tau_known ----
    tau_known = train.get('tau_known')
    if tau_known is None or not np.isfinite(tau_known):
        raise ValueError('fit_irf_sd_train: tau_known is empty. '
                         'Run estimate_tau_from_lopap first.')

    events_cell = []
    events_step_cell = []
    delay_exp = np.full(n_exp, np.nan)
    tmin_all = []
    tmax_all = []

    for i in range(n_exp):
        pack = exp_pack[i]
        y1 = _get_series(pack, 'y1', 'inst1')
        n = len(y1)

        if 'events' not in pack or not pack['events']:
            raise ValueError(f'fit_irf_sd_train: events empty for exp {i}.')

        pack_evs = pack['events']
        W = max(50, int(round(train['W'])))

        evs = []
        for k, pev in enumerate(pack_evs):
            t0 = pev['t0']
            etype = pev.get('type', pev.get('direction', 'rise'))
            a = max(0, t0 - W)
            b = min(n, t0 + W)
            evs.append({
                't0': t0, 'type': etype,
                'idx': np.arange(a, b, dtype=int),
                'reason': 'manifest', 'step_score': 1,
                'flat_pre': 0, 'flat_post': 0,
                'delta': 1, 'Lpre': 0, 'Lpost': 1,
                'sgn_dy': 1 if etype == 'rise' else -1,
            })

        events_cell.append(evs)
        events_step_cell.append(evs)
        delay_exp[i] = tau_known
        tmin_all.append(tau_known)

        # tmax estimation
        tmin_i, tmax_i = _estimate_tmin_tmax_from_events(y1, evs, train.get('tm'))
        if np.isfinite(tmax_i):
            tmax_all.append(tmax_i)

    # ---- global timing ----
    tmin_global = int(round(np.median(tmin_all)))
    if tmax_all:
        tmax_global = int(round(np.quantile(tmax_all, 0.8)))
    else:
        tmax_global = tmin_global + 500
    if tmax_global <= tmin_global:
        tmax_global = tmin_global + 500

    print(f'=== Global estimates: td={tmin_global}, tmax={tmax_global}, '
          f'Tmem={tmax_global - tmin_global} ===')

    delta_exp = delay_exp - tmin_global

    # ---- add windows + isolated selection ----
    events_iso_cell = []
    for i in range(n_exp):
        y1 = _get_series(exp_pack[i], 'y1', 'inst1')
        n = len(y1)
        evs_step = events_step_cell[i]
        if not evs_step:
            events_iso_cell.append([])
            continue
        evs_step = _add_event_windows(evs_step, n, tmin_global, tmax_global,
                                      train.get('win', {}))
        events_step_cell[i] = evs_step
        events_iso_cell.append(_select_isolated_events(evs_step, greedy=True))

    # ---- detail struct ----
    detail = {
        'events_step': events_step_cell,
        'events_iso': events_iso_cell,
        'tmin_global': tmin_global,
        'tmax_global': tmax_global,
        'delay_exp': delay_exp,
        'delta_exp': delta_exp,
        'delay_fixed': tmin_global,
    }

    delay_fixed = tmin_global

    # ---- bounds (phi = [tau_f, Pe_f, w, tau_s, Pe_s]) ----
    phi0 = np.array([100.0, 50.0, 0.80, 400.0, 5.0])
    lb_phi = np.array([30.0, 10.0, 0.70, 200.0, 0.5])
    ub_phi = np.array([250.0, 150.0, 0.95, 800.0, 15.0])

    cfg_train = cfg.get('train', {})
    if 'phi0' in cfg_train and len(cfg_train['phi0']) == 5:
        phi0 = np.array(cfg_train['phi0'], dtype=float)
    if 'lb' in cfg_train and len(cfg_train['lb']) == 6:
        lb_phi = np.array(cfg_train['lb'][1:6], dtype=float)
    if 'ub' in cfg_train and len(cfg_train['ub']) == 6:
        ub_phi = np.array(cfg_train['ub'][1:6], dtype=float)

    if np.any(lb_phi >= ub_phi):
        warnings.warn('Invalid bounds (lb >= ub), using defaults.')
        lb_phi = np.array([30.0, 10.0, 0.70, 200.0, 0.5])
        ub_phi = np.array([250.0, 150.0, 0.95, 800.0, 15.0])

    phi0 = np.clip(phi0, lb_phi, ub_phi)

    print(f'=== DUAL-IG: phi0=[tau_f={phi0[0]:.1f}, Pe_f={phi0[1]:.2f}, '
          f'w={phi0[2]:.3f}, tau_s={phi0[3]:.1f}, Pe_s={phi0[4]:.2f}] ===')

    # ---- optimiser options ----
    opt_cfg = cfg.get('opt', {}).get('fmincon', {})
    opt_kwargs = {
        'maxiter': opt_cfg.get('maxiter', 200),
        'ftol': opt_cfg.get('ftol', 1e-6),
        'gtol': opt_cfg.get('gtol', 1e-8),
        'disp': opt_cfg.get('disp', True),
    }

    # ---- base events ----
    use_isolated = train.get('use_isolated_events', False)
    events_base = events_iso_cell if use_isolated else events_step_cell

    train['delay_fixed'] = delay_fixed
    train['w_center_fixed'] = delay_fixed
    train['delta_exp'] = delta_exp

    bounds = list(zip(lb_phi, ub_phi))

    def obj_fn(phi):
        return _kernel_objective(phi, delay_fixed, exp_pack, twin, train,
                                 events_base, dt)

    # ====== Phase 0: Automatic parameter search (LHS) ======
    if train.get('auto_search', False):
        print('\n=== PHASE 0: Automatic parameter search ===')

        search_lb = np.array(cfg_train.get('search_lb', lb_phi), dtype=float)
        search_ub = np.array(cfg_train.get('search_ub', ub_phi), dtype=float)
        N_coarse = int(train.get('search_N_coarse', 500))
        N_topk = int(train.get('search_N_topk', 15))

        # Latin Hypercube sampling
        sampler = LatinHypercube(d=5, seed=42)
        lhs_unit = sampler.random(n=N_coarse)
        phi_samples = search_lb + lhs_unit * (search_ub - search_lb)

        fvals_coarse = np.full(N_coarse, np.inf)
        print(f'  Coarse search ({N_coarse} points)...', end='', flush=True)
        import time
        t_search = time.time()
        for j in range(N_coarse):
            try:
                fvals_coarse[j] = obj_fn(phi_samples[j])
            except Exception:
                pass
        print(f' done ({time.time() - t_search:.1f} s)')

        sort_idx = np.argsort(fvals_coarse)
        topk_idx = sort_idx[:min(N_topk, N_coarse)]

        # Quick refinement of top candidates
        search_bounds = list(zip(search_lb, search_ub))
        phi_refined = np.full((len(topk_idx), 5), np.nan)
        fval_refined = np.full(len(topk_idx), np.inf)

        for j, ti in enumerate(topk_idx):
            p0 = phi_samples[ti]
            try:
                res = minimize(obj_fn, p0, method='L-BFGS-B',
                               bounds=search_bounds,
                               options={'maxiter': 50, 'maxfun': 300,
                                        'ftol': 1e-5, 'gtol': 1e-7})
                phi_refined[j] = res.x
                fval_refined[j] = res.fun
            except Exception:
                pass

        best_idx = np.nanargmin(fval_refined)
        phi_best = phi_refined[best_idx]
        print(f'  Best: tau_f={phi_best[0]:.1f}, Pe_f={phi_best[1]:.2f}, '
              f'w={phi_best[2]:.3f}, tau_s={phi_best[3]:.1f}, '
              f'Pe_s={phi_best[4]:.2f}')
        phi0 = phi_best

        if train.get('search_auto_tighten', False):
            valid_r = np.isfinite(phi_refined[:, 0]) & np.isfinite(fval_refined)
            good = valid_r & (fval_refined < fval_refined[best_idx] * 2.0)
            if np.sum(good) < 3:
                good = valid_r
            phi_good = phi_refined[good]
            margin = 0.25
            lb_phi = np.maximum(search_lb, phi_good.min(axis=0) * (1 - margin))
            ub_phi = np.minimum(search_ub, phi_good.max(axis=0) * (1 + margin))
            lb_phi = np.minimum(lb_phi, phi_best * 0.9)
            ub_phi = np.maximum(ub_phi, phi_best * 1.1)
            bounds = list(zip(lb_phi, ub_phi))

    # ====== Pass 1: train ======
    print('\n=== Pass 1: Training ===')
    res1 = minimize(obj_fn, phi0, method='L-BFGS-B', bounds=bounds,
                    options=opt_kwargs)
    phi1 = res1.x
    fval1 = res1.fun
    theta1 = np.array([delay_fixed, *phi1])

    # ====== Pass 2: event selection + retrain ======
    events_selected = events_base
    do_select = train.get('do_select_topk', False)

    if do_select and train.get('two_pass', False):
        print('\n=== Event selection (top-K per exp) ===')
        events_scored = _score_events_by_fit(
            exp_pack, events_base, theta1, twin, train, dt)
        events_selected = _select_topk_events(events_scored, train)

        def obj_fn2(phi):
            return _kernel_objective(phi, delay_fixed, exp_pack, twin, train,
                                     events_selected, dt)

        print('\n=== Pass 2: Retrain with selected events ===')
        res2 = minimize(obj_fn2, phi1, method='L-BFGS-B', bounds=bounds,
                        options=opt_kwargs)
        phi_star = res2.x
        fval = res2.fun
        theta_star = np.array([delay_fixed, *phi_star])
        detail['selection_enabled'] = True
        detail['events_selected'] = events_selected
    else:
        theta_star = theta1
        fval = fval1
        detail['selection_enabled'] = False
        detail['events_selected'] = events_selected

    # ---- build final IRF ----
    alpha = train.get('anchor_alpha', 0.05)
    h_total, I, _ = irf_from_theta(theta_star, twin, dt, alpha)

    sd_star = {
        'theta': theta_star,
        'h_total': h_total,
        'I': I,
        't_window': twin,
        'dt': dt,
        'tmin': delay_fixed,
        'tmax': tmax_global,
        'alpha': alpha,
        'kernel_mode': 'dual_ig',
    }

    print(f'\nTRAIN DONE: theta=[delay {theta_star[0]:.1f}, '
          f'tau_f={theta_star[1]:.1f}, Pe_f={theta_star[2]:.2f}, '
          f'w={theta_star[3]:.3f}, tau_s={theta_star[4]:.1f}, '
          f'Pe_s={theta_star[5]:.2f}], fval={fval:.6g}')

    return sd_star, fval, detail
