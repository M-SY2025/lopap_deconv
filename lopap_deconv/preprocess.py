"""
Data preprocessing for LOPAP deconvolution.

Includes default configuration, data packing, and transport delay estimation.
"""

import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d


def cfg_default():
    """Return the default configuration dict for LOPAP IRF estimation.

    Returns
    -------
    dict
        Nested configuration with keys: Kbase, tau_init, reg, diag, opt.
    """
    cfg = {
        'Kbase': 5,
        'tau_init': 500,

        # regularization
        'reg': {
            'prepad': 0,
            'postpad': 0,
            'tailpad': 0,
            'lambda_end0': 0,
            'lambda1_smooth': 0.15,
            'lambda2_curv': 0.10,
            'lambda_align': 10,
            'lambda_beta': 1e-3,
            'lambda_c2': 0,
            'lambda_gamma': 0,
            'align_smooth': 31,
            'align_shift': 0,
            'align_use_w': True,
            'c2_use_w': False,
            'use_dir_ineq': False,
        },

        # diagnostics
        'diag': {
            'smooth_win': 31,
            'thrs': [0.10, 0.50, 0.90],
        },

        # optimization (scipy options)
        'opt': {
            'lsqlin': {
                'max_iter': 200,
                'tol': 1e-8,
            },
            'fmincon': {
                'maxiter': 200,
                'maxfev': 2000,
                'ftol': 1e-6,
                'gtol': 1e-8,
            },
        },
    }
    return cfg


def pack_lopap_only(inst1, tau, events=None, Kbase=5, mode='train', tmax_pad=None):
    """Pack LOPAP data for IRF training or deconvolution.

    Parameters
    ----------
    inst1 : array_like
        LOPAP concentration (uniform grid, 1-D).
    tau : int
        Transport delay in grid points.
    events : list of dict or None
        Event list with keys t0_idx, direction (from manifest).
    Kbase : int
        Number of baseline points for correction (0 = no correction).
    mode : str
        'train' or 'deconv'.
    tmax_pad : int or None
        Padding length for deconv mode (default: 2*tau).

    Returns
    -------
    dict
        Packed experiment data compatible with train/deconv functions.
    """
    inst1 = np.asarray(inst1, dtype=float).ravel()
    n_raw = len(inst1)
    tau = max(0, int(round(tau)))
    if tmax_pad is None:
        tmax_pad = 2 * tau

    # baseline correction
    if Kbase > 0:
        Kbase = max(1, min(Kbase, n_raw))
        bl = np.mean(inst1[:Kbase])
        inst1 = inst1 - bl

    if mode == 'train':
        pack = {
            'y1': inst1.copy(),
            'n': n_raw,
            'n_raw': n_raw,
            'tau': tau,
            'tau_i': tau,
            'prepad': 0,
            'prepad_base': 0,
            'tailpad': 0,
        }

        if events is not None and len(events) > 0:
            ev_out = []
            t0_locs = []
            for k, ev in enumerate(events):
                t0 = ev['t0_idx']
                t0 = max(0, min(n_raw - 1, t0))
                direction = ev['direction']

                W = 250
                a = max(0, t0 - W)
                b = min(n_raw, t0 + W)

                ev_out.append({
                    't0': t0,
                    'type': direction,
                    'idx': np.arange(a, b),
                })
                t0_locs.append(t0)

            pack['events'] = ev_out
            pack['event_locs'] = np.array(t0_locs)
            pack['K'] = len(ev_out)
        else:
            pack['events'] = []
            pack['event_locs'] = np.array([])
            pack['K'] = 0

        print(f"  [pack_lopap_only:train] n={n_raw}, tau={tau}, "
              f"{pack['K']} events")

    elif mode == 'deconv':
        if Kbase > 0:
            # WITH BASELINE: artificial padding
            prepad_base = tau
            n_total = prepad_base + tau + n_raw

            n_end = min(3, n_raw)
            head1 = np.mean(inst1[:n_end])
            inst1_padded = np.concatenate([
                np.full(prepad_base + tau, head1),
                inst1,
            ])

            pack = {
                'y1': inst1_padded,
                'n': n_total,
                'n_raw': n_raw,
                'tau': tau,
                'tau_i': tau,
                'prepad': prepad_base,
                'prepad_base': prepad_base,
                'tailpad': tau,
                'events': [],
                'event_locs': np.array([]),
                'K': 0,
            }

            print(f"  [pack_lopap_only:deconv:padded] n_raw={n_raw}, "
                  f"prepad={prepad_base}, tau={tau}, n_total={n_total}")

        else:
            # NO BASELINE: pad with actual signal level
            prepad_len = int(round(tmax_pad))
            tailpad_len = int(round(tmax_pad))

            n_avg = max(3, min(30, int(round(n_raw * 0.01))))
            head_val = np.median(inst1[:n_avg])
            tail_val = np.median(inst1[-n_avg:])

            inst1_padded = np.concatenate([
                np.full(prepad_len, head_val),
                inst1,
                np.full(tailpad_len, tail_val),
            ])
            n_total = len(inst1_padded)

            pack = {
                'y1': inst1_padded,
                'n': n_total,
                'n_raw': n_raw,
                'tau': tau,
                'tau_i': tau,
                'prepad': prepad_len,
                'prepad_base': prepad_len,
                'tailpad': tailpad_len,
                'events': [],
                'event_locs': np.array([]),
                'K': 0,
            }

            print(f"  [pack_lopap_only:deconv:no-baseline] n_raw={n_raw}, "
                  f"prepad={prepad_len}(val={head_val:.3f}), "
                  f"tailpad={tailpad_len}(val={tail_val:.3f}), "
                  f"n_total={n_total}")
    else:
        raise ValueError(f'Unknown mode "{mode}". Use "train" or "deconv".')

    return pack


def estimate_tau_from_lopap(inst1, events, opts=None):
    """Estimate transport delay tau from LOPAP signal + known t0.

    Uses a dual-method approach (step-fraction + derivative) and takes
    the more conservative (later) onset estimate.

    Parameters
    ----------
    inst1 : array_like
        LOPAP concentration (uniform grid).
    events : list of dict
        Each with 't0_idx' (0-based) and 'direction' ('rise' or 'fall').
    opts : dict, optional
        Override defaults for smooth_win, search_max, alpha, beta, etc.

    Returns
    -------
    tau_est : int
        Median tau estimate across all events (in grid points).
    diag : dict
        Per-event diagnostic details.
    """
    inst1 = np.asarray(inst1, dtype=float).ravel()
    n = len(inst1)

    # defaults
    defaults = {
        'smooth_win': 31,
        'search_max': 2500,
        'alpha': 0.05,
        'beta': 0.20,
        'onset_persist': 10,
        'confirm_win': 300,
        'preA': 600,
        'preB': 100,
        'tau_min': 100,
        'tau_max': 900,
    }
    if opts is not None:
        defaults.update({k: v for k, v in opts.items() if v is not None})
    p = defaults

    # smooth
    if p['smooth_win'] > 1:
        y1s = uniform_filter1d(inst1, size=p['smooth_win'], mode='nearest')
    else:
        y1s = inst1.copy()

    nEv = len(events)
    tau_per_event = np.full(nEv, np.nan)
    ev_diag = []

    for k in range(nEv):
        t0 = events[k]['t0_idx']
        is_rise = events[k]['direction'] == 'rise'

        ed = {'t0_idx': t0, 'direction': events[k]['direction'],
              'valid': False, 'flag': ''}

        # ---- pre-step baseline ----
        pre_lo = max(0, t0 - int(round(p['preA'])))
        pre_hi = max(0, t0 - int(round(p['preB'])))

        if pre_hi <= pre_lo + 30:
            ed['flag'] = 'insufficient_baseline'
            warnings.warn(f"Event {k} (t0={t0}): only {pre_hi - pre_lo} "
                          f"baseline points, skipping.")
            ev_diag.append(ed)
            continue

        Lpre = np.median(y1s[pre_lo:pre_hi + 1])

        # ---- post-step level ----
        search_end = min(n, t0 + int(round(p['search_max'])))
        seg_search = y1s[t0:search_end + 1]

        if len(seg_search) < 50:
            ed['flag'] = 'insufficient_search_data'
            ev_diag.append(ed)
            continue

        if is_rise:
            Lpost = np.percentile(seg_search, 90)
        else:
            Lpost = np.percentile(seg_search, 10)

        # ---- step amplitude ----
        delta = (Lpost - Lpre) if is_rise else (Lpre - Lpost)

        ed['Lpre'] = Lpre
        ed['Lpost'] = Lpost
        ed['delta'] = delta

        if delta <= 0:
            ed['flag'] = f'non_positive_step (delta={delta:.4f})'
            ev_diag.append(ed)
            continue

        # ---- thresholds ----
        if is_rise:
            thr_on = Lpre + p['alpha'] * delta
            thr_cfm = Lpre + p['beta'] * delta
        else:
            thr_on = Lpre - p['alpha'] * delta
            thr_cfm = Lpre - p['beta'] * delta

        ed['thr_on'] = thr_on
        ed['thr_cfm'] = thr_cfm

        print(f"    Ev{k} [{events[k]['direction']}]: Lpre={Lpre:.4f}, "
              f"Lpost={Lpost:.4f}, delta={delta:.4f}, "
              f"thr_on={thr_on:.4f}, thr_cfm={thr_cfm:.4f}")

        # ---- search for onset: Method 1 (step-fraction) ----
        s0 = min(n - 1, t0 + 1)
        s1 = min(n - 1, t0 + int(round(p['search_max'])))
        persist = int(round(p['onset_persist']))

        if s1 <= s0 + persist:
            ed['flag'] = 'no_search_range'
            ev_diag.append(ed)
            continue

        cand_frac = np.nan
        run = 0
        for t in range(s0, s1 + 1):
            if is_rise:
                ok = y1s[t] >= thr_on
            else:
                ok = y1s[t] <= thr_on
            if ok:
                run += 1
                if run >= persist:
                    cand_frac = t - persist + 1
                    break
            else:
                run = 0

        # confirm with beta threshold
        if np.isfinite(cand_frac):
            t2 = min(n - 1, int(cand_frac) + int(round(p['confirm_win'])))
            seg = y1s[int(cand_frac):t2 + 1]
            if is_rise:
                confirmed = np.any(seg >= thr_cfm)
            else:
                confirmed = np.any(seg <= thr_cfm)
            if not confirmed:
                cand_frac = np.nan

        # ---- Method 2 (derivative) ----
        dy1s = np.diff(y1s, append=0)

        db0 = max(0, t0 - 1200)
        db1 = max(0, t0 - 200)
        if db1 <= db0 + 50:
            db0 = max(0, t0 - 800)
            db1 = max(0, t0 - 100)

        base_dy = dy1s[db0:db1 + 1]
        sig_dy = 1.4826 * np.median(np.abs(base_dy - np.median(base_dy))) + np.finfo(float).eps
        deriv_k = 3.0
        thr_dy = deriv_k * sig_dy

        cand_deriv = np.nan
        z = dy1s[s0:s1 + 1].copy()
        if not is_rise:
            z = -z

        run = 0
        deriv_persist = max(15, persist)
        for t_idx in range(len(z)):
            if z[t_idx] > thr_dy:
                run += 1
                if run >= deriv_persist:
                    cand_deriv = s0 + t_idx - deriv_persist + 1
                    break
            else:
                run = 0

        # ---- combine: take the LATER onset ----
        candidates = [c for c in [cand_frac, cand_deriv] if np.isfinite(c)]

        if not candidates:
            ed['flag'] = 'no_onset_detected'
            warnings.warn(f"Event {k} (t0={t0}): no onset detected "
                          f"within {p['search_max']} s.")
            ev_diag.append(ed)
            continue

        cand = int(max(candidates))
        onset_idx = cand
        tau_raw = cand - t0

        ed['onset_idx'] = onset_idx
        ed['tau_raw'] = tau_raw

        print(f"    Ev{k}: onset -> tau={tau_raw} s")

        # plausibility check
        if tau_raw < p['tau_min'] or tau_raw > p['tau_max']:
            ed['valid'] = True
            ed['flag'] = f"tau={tau_raw} outside [{p['tau_min']},{p['tau_max']}]"
            warnings.warn(f"Event {k}: tau={tau_raw} s outside plausible range.")
        else:
            ed['valid'] = True
            ed['flag'] = 'ok'

        tau_per_event[k] = tau_raw
        ev_diag.append(ed)

    # ---- aggregate ----
    valid_tau = tau_per_event[np.isfinite(tau_per_event)]
    if len(valid_tau) == 0:
        raise RuntimeError(f"No valid tau estimates from {nEv} events.")

    tau_est = int(round(np.median(valid_tau)))

    diag = {
        'tau_per_event': tau_per_event,
        'tau_est': tau_est,
        'tau_mean': float(np.mean(valid_tau)),
        'tau_std': float(np.std(valid_tau)),
        'n_valid': len(valid_tau),
        'n_total': nEv,
        'ev': ev_diag,
    }

    print(f"  [estimate_tau] {len(valid_tau)}/{nEv} events valid | "
          f"tau: median={tau_est}, mean={diag['tau_mean']:.0f}, "
          f"std={diag['tau_std']:.1f} s")

    return tau_est, diag
