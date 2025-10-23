# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:07:15 2025

@author: Hoffmann
"""
import pandas as pd
from pyrsktools import RSK


def read_logger_data(file_str):
    # RBR Handling
    if '.rsk' in file_str:
        with RSK(file_str) as rsk:
            raw = rsk.copy()
        raw.readdata()

        time = raw.data['timestamp']
        temp = raw.data['temperature']
        try:
            p = raw.data['pressure']
        except:
            p = -999

        return time, temp, p, raw.instrument.model

    # HOBO Handling
    if '.hobo' in file_str:
        return -999, -999, -999, 'hobo'


def subtract_reference(df, reference_serial, tolerance_hours=12,
                       sercol='Serial', tcol='logt', Tcol='logT', pcol='logp',
                       NNcol='DepthNN', namcol='GWM'):
    # Extract the reference dataset
    ref_row = df[df[sercol] == reference_serial].iloc[0]

    ref_df = pd.DataFrame({'time': ref_row[tcol], 'pref': ref_row[pcol]})
    results = []

    for _, row in df.iterrows():
        if not isinstance(row[pcol], int) and row[sercol] != reference_serial:

            # Build current dataset
            dfi = pd.DataFrame({'time': row[tcol],
                                'temp': row[Tcol],
                                'p': row[pcol],
                                'dNN': row[NNcol]})

            # Align by exact timestamps (inner join)
            merged = pd.merge_asof(dfi, ref_df, on='time', direction='nearest',
                                   tolerance=pd.Timedelta(hours=tolerance_hours))

            # Subtract pressures
            merged['p_diff'] = merged['p'] - merged['pref']
            merged['GWL_NN'] = merged['dNN'] - merged['p_diff']
            if row[namcol] == 'GWM6':
                print(merged['p_diff'])
            merged = merged[merged['p_diff'] > 1]

            # Store result
            results.append({
                'ref': row[sercol],
                'name': row[namcol],
                'time': merged['time'].tolist(),
                'temp': merged['temp'].tolist(),
                'p_diff': merged['p_diff'].tolist(),
                'GWL_NN': merged['GWL_NN'].tolist()
            })

    return pd.DataFrame(results)
