# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:25:00 2025

@author: DevHov
"""


import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import numpy as np


def plot_gw(ax, df, xcol, ycol, ylabel, xlabel='Date', namcol='GWM'):
    for _, row in df.iterrows():
        if isinstance(row[ycol], list):
            print(len(row[xcol]), len(row[ycol]))
            ax.plot(row[xcol], row[ycol], label=row[namcol])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def plot_temperature_profiles(df, cmap='turbo', figsize=(9, 6), save=False, outdir='./plots'):
    """
    Generate a time–depth–temperature colormap for each monitoring well (GWM)
    from logger data stored in a pandas DataFrame.

    Expected DataFrame columns:
        ['Serial', 'type', 'GWM', 'DepthNN', 'GWL_NN', 'logt', 'logT', 'logp']

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing logger data.
    cmap : str
        Matplotlib colormap for plotting (default: 'turbo').
    figsize : tuple
        Figure size for each plot.
    save : bool
        If True, plots will be saved to the specified directory instead of shown.
    outdir : str
        Directory to save plots when save=True.
    """

    import user_definitions as ud
    import functions.data_io as fdo

    # 1. Filter out loggers that do not measure temperature
    df_temp = df[~df['logT'].isin([-999])]

    if df_temp.empty:
        msg = "No temperature loggers found (all logT = -999)."
        fdo.log_message(msg, 'error')
        return

    # 2. Iterate over each monitoring well
    for gwm, group in df_temp.groupby('GWM'):
        print('GWM:', gwm)
        print('Group:', group)
        if group['Serial'].iloc[0] == ud.barometric_reference_logger:
            continue
        else:
            msg = f"Generating temperature profile for well: {gwm}"
            fdo.log_message(msg)

        all_times = []
        all_depths = []

        # Collect all timestamps for time grid generation
        for _, row in group.iterrows():
            times = np.array(row['logt'])
            temps = np.array(row['logT'], dtype=float)
            depth = row['DepthNN']

            # Remove NaN values
            valid = ~np.isnan(temps)
            times = times[valid]
            temps = temps[valid]

            if len(times) == 0:
                continue

            all_times.extend(times)
            all_depths.append(depth)

        if len(all_times) == 0:
            msg = f"No valid temperature data for well {gwm}."
            fdo.log_message(msg, 'error')
            continue

        # 3. Define unique time and depth grids
        time_unique = np.sort(pd.to_datetime(pd.Series(all_times).unique()))
        depth_unique = np.sort(group['DepthNN'].unique())

        temp_matrix = np.full((len(depth_unique), len(time_unique)), np.nan)

        # 4. Fill temperature matrix
        for i, depth in enumerate(depth_unique):
            sel = group[group['DepthNN'] == depth].iloc[0]
            t = pd.to_datetime(sel['logt'])
            T = pd.Series(sel['logT'], index=t).astype(
                float).dropna().sort_index()

            if T.empty:
                continue

            # Interpolate to the common time grid
            T_interp = T.reindex(time_unique, method='nearest')
            temp_matrix[i, :] = T_interp.values

        # 5. Plot the colormap
        fig, ax = plt.subplots(figsize=figsize)
        X, Y = np.meshgrid(time_unique, depth_unique)

        pcm = ax.pcolormesh(X, Y, temp_matrix, shading='auto', cmap=cmap)
        ax.invert_yaxis()
        ax.set_xlabel("Time")
        ax.set_ylabel("Depth [m]")
        ax.set_title(f"Temperature evolution in well {gwm}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        plt.colorbar(pcm, ax=ax, label='Temperature [°C]')
        plt.tight_layout()

        if save:
            import os
            os.makedirs(outdir, exist_ok=True)
            outfile = f"{outdir}/temperature_profile_{gwm}.png"
            plt.savefig(outfile, dpi=200)
            fdo.log_message(f"Saved plot to: {outfile}")
            plt.close(fig)
        else:
            plt.show()
