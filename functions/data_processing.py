# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 16:05:50 2025

@author: DevHov
"""
import pandas as pd
import numpy as np


def expand_row_to_dataframe(row):
    """Expand a single row containing list columns into a full DataFrame."""
    list_cols = [col for col in row.index if isinstance(row[col], list)]

    # If no list columns exist, just return the row as a single-row DataFrame
    if not list_cols:
        # log_message(f'Row for dataset {row} is 1-Dimensional', 'warning')
        return pd.DataFrame({col: [row[col]] for col in row.index})

    # Otherwise, expand based on the longest list
    max_len = max(len(row[col]) for col in list_cols)
    df_row = pd.DataFrame({
        col: row[col] if isinstance(row[col], list) else [row[col]] * max_len
        for col in row.index
    })
    return df_row.sort_values('logt')


def split_reference(df, reference_serial, sercol='Serial', tcol='logt', Tcol='logT', pcol='logp'):
    # returns two dataframes 1st with the structure of the old df containing
    # all datasets besides the reference 2nd is the reference with cols for
    # time, pressure and temperature
    ref_row = df[df[sercol] == reference_serial].iloc[0]
    ref_df = pd.DataFrame({'logt': ref_row[tcol],
                           'logT': ref_row[Tcol],
                          'logp': ref_row[pcol]})
    df_gw = df[df[sercol] != reference_serial]

    return df_gw, ref_df


def get_t0(df, pcol='logp', tcol='logt', namcol='GWM'):
    """
    Determine t0 (time of submersion) for each station (GWM)
    based on the logger that records valid pressure data.
    Returns a dict mapping GWM -> t0.
    """
    t0_dict = {}

    for name, group in df.groupby(namcol):
        # Find one row with valid pressure data
        valid = group[~group[pcol].apply(lambda x: isinstance(x, int))]
        if valid.empty:
            continue

        # Use first valid logger for this station
        row = valid.iloc[0]
        pressures = pd.Series(row[pcol])
        times = pd.Series(row[tcol])

        diff_idx = pressures.diff().idxmax()
        t0_dict[name] = times.iloc[diff_idx]

    return t0_dict

# %% test


def get_reference_data(df, reference_serial, sercol='Serial', tcol='logt', pcol='logp'):
    """Extract timestamp and pressure columns of the reference logger."""
    ref_row = df[df[sercol] == reference_serial].iloc[0]
    ref_df = pd.DataFrame({
        tcol: ref_row[tcol],
        pcol: ref_row[pcol]
    })
    return ref_df


def align_datasets(df_row, ref_df, tolerance_hours=12,
                   tcol='logt', Tcol='logT', pcol='logp',
                   NNcol='DepthNN', mancol='GWL_NN'):
    """Align timestamps of the current logger with the reference dataset using merge_asof."""
    # Build dataframe for the current logger
    dfi = expand_row_to_dataframe(df_row)

    # Sort and drop NaNs to ensure merge_asof works properly
    dfi = dfi.dropna(subset=[tcol]).sort_values(tcol)
    ref_df = ref_df.dropna(subset=[tcol]).sort_values(tcol)

    # Ensure consistent datetime precision
    dfi[tcol] = pd.to_datetime(
        dfi[tcol], errors='coerce').astype('datetime64[ns]')
    ref_df[tcol] = pd.to_datetime(
        ref_df[tcol], errors='coerce').astype('datetime64[ns]')

    # Perform nearest time alignment within the specified tolerance
    merged = pd.merge_asof(
        dfi, ref_df, on=tcol, direction='nearest',
        tolerance=pd.Timedelta(hours=tolerance_hours)
    )
    return merged


def compute_pressure_difference(df, z_corr_idx, pcol='logp'):
    """Compute pressure difference and adjusted groundwater levels."""
    # Compute pressure difference
    df['p_diff'] = df[pcol+'_x'] - df[pcol+'_y']

    # Filter physically meaningful values, return a copy explicitly
    # df = df.loc[df['p_diff'] > 1].copy()

    df.loc[:, 'log_gwl'] = df['GWL_NN'] - df['p_diff']
    df.loc[:, 'log_gwl'] = df['log_gwl'] + df['p_diff'].iloc[z_corr_idx]

    return df


def subtract_reference(df, reference_serial, tolerance_hours=12,
                       sercol='Serial', tcol='logt', Tcol='logT', pcol='logp',
                       NNcol='DepthNN', namcol='GWM', mancol='GWL_NN'):
    """
    Subtracts the barometric reference pressure from all other loggers
    to compute corrected groundwater levels.
    """
    ref_df = get_reference_data(df, reference_serial, sercol, tcol, pcol)
    results = []

    for _, row in df.iterrows():
        # Skip reference and entries without valid pressure data
        if row[sercol] == reference_serial or isinstance(row[pcol], int):
            continue

        merged = align_datasets(
            row, ref_df, tolerance_hours, tcol, Tcol, pcol, NNcol, mancol)
        merged = compute_pressure_difference(merged)

        results.append({
            'ref': reference_serial,
            'name': row[namcol],
            'time': merged['time'].tolist(),
            'temp': merged['temp'].tolist(),
            'p_diff': merged['p_diff'].tolist(),
            'GWL_NN': merged['GWL_NN'].tolist()
        })

    return pd.DataFrame(results)


def clean_dataset(df, reference_serial, tolerance_hours=12,
                  sercol='Serial', tcol='logt', Tcol='logT', pcol='logp',
                  NNcol='DepthNN', namcol='GWM', mancol='GWL_NN'):
    """
    Cleans datasets by trimming each station's data to start at its submersion time (t0)
    and computing pressure differences, keeping the 'wide' format:
    one row per logger, lists for each column.
    """
    ref_df = get_reference_data(df, reference_serial, sercol, tcol, pcol)
    results = []

    # Step 1: determine t0 for each station
    t0_dict = get_t0(df, pcol, tcol, namcol)

    for _, row in df.iterrows():
        # Skip reference
        if row[sercol] == reference_serial:
            continue

        name = row[namcol]
        t0 = t0_dict.get(name, None)
        if t0 is None:
            continue

        # Align with reference logger (long-format)
        df_sync = align_datasets(
            row, ref_df, tolerance_hours, tcol, Tcol, pcol, NNcol, mancol)

        # Trim lists according to t0
        mask_t0 = df_sync[tcol] > t0
        logt = df_sync.loc[mask_t0, tcol].tolist()
        logT = df_sync.loc[mask_t0, Tcol].tolist()

        mask_gwl = np.ones(len(logt), dtype=bool)

        if not isinstance(row[pcol], int):
            z_corr_idx = mask_t0.idxmax()
            df_sync = compute_pressure_difference(df_sync, z_corr_idx, pcol)
            p_diff = df_sync.loc[mask_t0, 'p_diff'].tolist()
            log_gwl = df_sync.loc[mask_t0, 'log_gwl'].tolist()

            median_gwl = np.median(
                df_sync.loc[mask_t0, 'log_gwl'].replace(-999, np.nan).dropna())

            threshold = 1
            mask_gwl = np.array(log_gwl < median_gwl + threshold)
            log_gwl = np.array(log_gwl)[mask_gwl].tolist()
            p_diff = np.array(p_diff)[mask_gwl].tolist()

        else:
            p_diff = -999
            log_gwl = -999

        logt = np.array(logt)[mask_gwl].tolist()
        logT = np.array(logT)[mask_gwl].tolist()

        # Keep other columns from original row
        data_i = {
            sercol: row[sercol],
            'type': row['type'],
            namcol: name,
            NNcol: row[NNcol],
            mancol: row[mancol],
            tcol: logt,
            Tcol: logT,
            'p_diff': p_diff,
            'log_gwl': log_gwl
        }
        results.append(data_i)

    return pd.DataFrame(results)
