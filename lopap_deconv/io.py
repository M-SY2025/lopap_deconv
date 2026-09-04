"""
Data I/O for LOPAP deconvolution.

Functions for reading raw LOPAP data files, training manifests,
and test file lists.
"""

import os
import warnings
import numpy as np
import pandas as pd


def read_lopap_raw(filepath, dt_out=1):
    """Read a single LOPAP raw data file.

    Parameters
    ----------
    filepath : str
        Path to .csv or .xlsx file.
        Column 1: datetime, Column 2: concentration.
    dt_out : int
        Output grid interval in seconds.
        1 = 1 s grid (default). 0 = native sampling (no interpolation).

    Returns
    -------
    dict with keys:
        t_datetime, conc_raw, t_sec, conc, t0_datetime, dt_median, dt_out, filepath
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(filepath, header=None)
    elif ext in ('.xls', '.xlsx'):
        df = pd.read_excel(filepath, header=None)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .xls, or .xlsx.")

    # ---- skip header row if first cell is non-date text ----
    first_val = df.iloc[0, 0]
    if isinstance(first_val, str) and not pd.api.types.is_numeric_dtype(type(first_val)):
        try:
            pd.to_datetime(first_val)
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

    # ---- parse datetime column ----
    col1 = df.iloc[:, 0]
    if pd.api.types.is_datetime64_any_dtype(col1):
        dt_arr = pd.to_datetime(col1)
    elif pd.api.types.is_numeric_dtype(col1):
        # Excel serial date
        dt_arr = pd.to_datetime(col1, unit='D', origin='1899-12-30')
    else:
        dt_arr = pd.to_datetime(col1, format='mixed', dayfirst=False)

    # ---- parse concentration column ----
    conc_raw = pd.to_numeric(df.iloc[:, 1], errors='coerce').values.astype(float)

    # ---- remove NaN rows ----
    valid = dt_arr.notna() & np.isfinite(conc_raw)
    dt_arr = dt_arr[valid].reset_index(drop=True)
    conc_raw = conc_raw[valid]

    # ---- sort by time ----
    sort_idx = dt_arr.argsort()
    dt_arr = dt_arr.iloc[sort_idx].reset_index(drop=True)
    conc_raw = conc_raw[sort_idx]

    # ---- convert to seconds from first sample ----
    t0_datetime = dt_arr.iloc[0]
    t_sec_raw = (dt_arr - t0_datetime).dt.total_seconds().values

    # ---- aggregate duplicate timestamps ----
    df_agg = pd.DataFrame({'t': t_sec_raw, 'c': conc_raw})
    df_agg = df_agg.groupby('t', as_index=False).mean()
    t_sec_uniq = df_agg['t'].values
    conc_uniq = df_agg['c'].values

    # ---- determine output grid ----
    dt_native = np.median(np.diff(t_sec_uniq))
    if dt_out <= 0:
        dt_out = int(round(dt_native))

    # ---- interpolate to uniform grid ----
    t_sec = np.arange(0, int(t_sec_uniq[-1]) + 1, dt_out, dtype=float)
    conc = np.interp(t_sec, t_sec_uniq, conc_uniq)

    print(f"  [read_lopap_raw] {filepath}: {len(conc_raw)} raw samples, "
          f"dt_median={dt_native:.1f} s, -> {len(t_sec)} pts (dt={dt_out} s)")

    return {
        't_datetime': dt_arr,
        'conc_raw': conc_raw,
        't_sec': t_sec,
        'conc': conc,
        't0_datetime': t0_datetime,
        'dt_median': dt_native,
        'dt_out': dt_out,
        'filepath': filepath,
    }


def _parse_datetime_flex(series):
    """Parse datetime strings with flexible format detection."""
    return pd.to_datetime(series, format='mixed', dayfirst=False)


def read_lopap_manifest(manifest_path):
    """Read experiment manifest and load LOPAP data.

    Manifest format (CSV or Excel, first row is header)::

        file,          t0,                  direction
        exp1.csv,      2024/8/7 16:54:45,   rise
        exp1.csv,      2024/8/7 17:30:00,   fall

    Parameters
    ----------
    manifest_path : str
        Path to the manifest file.

    Returns
    -------
    list of dict
        One dict per unique file, with keys:
        filepath, inst1, t_sec, t0_datetime, events, raw
    """
    ext = os.path.splitext(manifest_path)[1].lower()
    if ext in ('.xls', '.xlsx'):
        T = pd.read_excel(manifest_path)
    else:
        T = pd.read_csv(manifest_path)

    T.columns = T.columns.str.lower().str.strip()
    assert 'file' in T.columns, 'Manifest must have "file" column.'
    assert 't0' in T.columns, 'Manifest must have "t0" column.'
    assert 'direction' in T.columns, 'Manifest must have "direction" column.'

    files = T['file'].str.strip()
    dir_str = T['direction'].str.strip().str.lower()
    t0_dt = _parse_datetime_flex(T['t0'].astype(str).str.strip())

    for i, d in enumerate(dir_str):
        if d not in ('rise', 'fall'):
            raise ValueError(f'Row {i}: direction must be "rise" or "fall", got "{d}".')

    # resolve relative paths
    mdir = os.path.dirname(manifest_path) or '.'
    resolved_files = []
    for i, f in enumerate(files):
        if os.path.isfile(f):
            resolved_files.append(f)
        else:
            f_rel = os.path.join(mdir, f)
            if os.path.isfile(f_rel):
                resolved_files.append(f_rel)
            else:
                raise FileNotFoundError(
                    f'Row {i}: file not found: "{f}" (also tried "{f_rel}").')

    # group by unique file
    unique_files = list(dict.fromkeys(resolved_files))  # preserves order
    group_map = {uf: [] for uf in unique_files}
    for i, f in enumerate(resolved_files):
        group_map[f].append(i)

    print(f"=== read_lopap_manifest: {len(T)} rows -> {len(unique_files)} experiments ===")

    exps = []
    for g, fpath in enumerate(unique_files):
        rows = group_map[fpath]
        raw = read_lopap_raw(fpath)

        events = []
        for r in rows:
            dt_sec = (t0_dt.iloc[r] - raw['t0_datetime']).total_seconds()
            if dt_sec < 0:
                warnings.warn(
                    f"Exp {g} event: t0 is before file start by {-dt_sec:.0f} s. Clamping to 0.")
                dt_sec = 0
            if dt_sec > raw['t_sec'][-1]:
                warnings.warn(
                    f"Exp {g} event: t0 is after file end by "
                    f"{dt_sec - raw['t_sec'][-1]:.0f} s.")

            events.append({
                't0_abs': t0_dt.iloc[r],
                't0_sec': dt_sec,
                't0_idx': int(round(dt_sec)),  # 0-based index into 1 s grid
                'direction': dir_str.iloc[r],
            })

        # sort events by time
        events.sort(key=lambda e: e['t0_sec'])

        exp = {
            'filepath': fpath,
            'inst1': raw['conc'].copy(),
            't_sec': raw['t_sec'].copy(),
            't0_datetime': raw['t0_datetime'],
            'events': events,
            'raw': raw,
        }
        exps.append(exp)

        ev_info = '  '.join(
            f"|  ev{k}: t0={e['t0_sec']:.0f}s ({e['direction']})"
            for k, e in enumerate(events))
        print(f"  Exp {g} [{fpath}]: {len(raw['conc'])} pts, "
              f"{len(events)} events  {ev_info}")

    print(f"=== Done: {len(exps)} experiments loaded ===")
    return exps


def read_lopap_test_list(list_path, dt_out=1):
    """Read a list of LOPAP files for deconvolution.

    The list file is a CSV with a single column "file"::

        file
        field_day1.xlsx
        field_day2.xlsx

    Parameters
    ----------
    list_path : str
        Path to the file list CSV.
    dt_out : int
        Output grid interval in seconds (default=1).

    Returns
    -------
    list of dict
        One dict per file, with keys:
        filepath, inst1, t_sec, t0_datetime, dt_out, raw
    """
    ext = os.path.splitext(list_path)[1].lower()
    if ext in ('.xls', '.xlsx'):
        T = pd.read_excel(list_path)
    else:
        T = pd.read_csv(list_path)

    T.columns = T.columns.str.lower().str.strip()
    assert 'file' in T.columns, 'Test list must have a "file" column.'

    files = T['file'].str.strip().tolist()

    # resolve relative paths
    ldir = os.path.dirname(list_path) or '.'
    for i, f in enumerate(files):
        if not os.path.isfile(f):
            f_rel = os.path.join(ldir, f)
            if os.path.isfile(f_rel):
                files[i] = f_rel
            else:
                raise FileNotFoundError(
                    f'Row {i}: file not found: "{f}" (also tried "{f_rel}").')

    print(f"=== read_lopap_test_list: {len(files)} files ===")

    exps = []
    for i, fpath in enumerate(files):
        raw = read_lopap_raw(fpath, dt_out)
        exp = {
            'filepath': fpath,
            'inst1': raw['conc'].copy(),
            't_sec': raw['t_sec'].copy(),
            't0_datetime': raw['t0_datetime'],
            'dt_out': raw['dt_out'],
            'raw': raw,
        }
        exps.append(exp)
        print(f"  Exp {i} [{fpath}]: {len(raw['conc'])} pts "
              f"({raw['t_sec'][-1]/60:.0f} min)")

    print(f"=== Done: {len(exps)} test experiments loaded ===")
    return exps
