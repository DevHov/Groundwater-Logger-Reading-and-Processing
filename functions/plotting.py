# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 17:25:00 2025

@author: DevHov
"""


import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os


def plot_gw(ax, df, xcol, ycol, ylabel, xlabel='Date', namcol='GWM'):
    for _, row in df.iterrows():
        if isinstance(row[ycol], list):
            ax.plot(row[xcol], row[ycol], label=row[namcol])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def plot_temperature_profiles(df, cmap='turbo', figsize=(9, 6), save=False, outdir='./plots',
                              depth_interp_resolution=0.05):
    """
    Generate a time–depth–temperature colormap for each monitoring well (GWM),
    interpolating both in time and between logger depths.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataset with columns ['Serial','type','GWM','DepthNN','logt','logT'].
    cmap : str
        Matplotlib colormap for plotting.
    figsize : tuple
        Figure size for each plot.
    save : bool
        If True, save plots to 'outdir'.
    outdir : str
        Directory to save plots.
    depth_interp_resolution : float
        Step size [m] for vertical interpolation grid.
    """

    # --- 1. Filter out invalid temperature loggers
    df_temp = df[~df['logT'].isin([-999])]

    if df_temp.empty:
        print("No temperature loggers found.")
        return

    # --- 2. Iterate wells
    for gwm, group in df_temp.groupby('GWM'):
        all_times = []
        all_depths = []

        # collect all time points
        for _, row in group.iterrows():
            times = np.array(pd.to_datetime(row['logt']))
            temps = np.array(row['logT'], dtype=float)
            depth = row['DepthNN']
            valid = ~np.isnan(temps)
            times = times[valid]
            temps = temps[valid]
            if len(times) == 0:
                continue
            all_times.extend(times)
            all_depths.append(depth)

        if len(all_times) == 0:
            print(f"No valid data for {gwm}")
            continue

        # --- 3. Define uniform time and depth grid
        time_unique = np.sort(pd.to_datetime(pd.Series(all_times).unique()))
        depth_unique = np.sort(group['DepthNN'].unique())

        # optional: finer vertical grid
        depth_fine = np.arange(depth_unique.min(), depth_unique.max() + depth_interp_resolution,
                               depth_interp_resolution)

        temp_matrix = np.full((len(depth_unique), len(time_unique)), np.nan)

        # --- 4. Fill temperature matrix for each depth
        for i, depth in enumerate(depth_unique):
            sel = group[group['DepthNN'] == depth].iloc[0]
            t = pd.to_datetime(sel['logt'])
            T = pd.Series(sel['logT'], index=t).astype(
                float).dropna().sort_index()
            if T.empty:
                continue
            # interpolate to common time grid (nearest)
            T_interp = T.reindex(time_unique, method='nearest')
            temp_matrix[i, :] = T_interp.values

        # --- 5. Interpolate vertically (depth axis)
        temp_interp = np.full((len(depth_fine), len(time_unique)), np.nan)
        for j in range(len(time_unique)):
            col = temp_matrix[:, j]
            valid = ~np.isnan(col)
            if valid.sum() >= 2:
                fz = interp1d(depth_unique[valid], col[valid],
                              kind='linear', bounds_error=False, fill_value=np.nan)
                temp_interp[:, j] = fz(depth_fine)

        # --- 6. Plot colormap
        fig, ax = plt.subplots(figsize=figsize)
        X, Y = np.meshgrid(time_unique, depth_fine)
        pcm = ax.pcolormesh(X, Y, temp_interp, shading='auto', cmap=cmap)

        ax.invert_yaxis()
        ax.set_xlabel("Time")
        ax.set_ylabel("Depth [m]")
        ax.set_title(f"Temperature evolution in well {gwm}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.invert_yaxis()
        fig.autofmt_xdate()
        plt.colorbar(pcm, ax=ax, label='Temperature [°C]')
        plt.tight_layout()

        if save:
            os.makedirs(outdir, exist_ok=True)
            outfile = f"{outdir}/temperature_profile_{gwm}.png"
            plt.savefig(outfile, dpi=200)
            print(f"Saved plot to {outfile}")
            plt.close(fig)
        else:
            plt.show()


def plot_time_depth_scatter(df, color_by='serial', figsize=(10, 6), save=False, outdir='./plots', alpha=0.7, s=10):
    """
    Scatter plot of timestamp vs depth for each GWM (monitoring well).
    Useful to inspect time coverage and alignment of loggers inside each well.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (output of clean_dataset), expected columns:
        ['Serial', 'GWM', 'DepthNN', 'logt', 'logT', ...]
    color_by : str
        'serial' (default) colors points by serial number (discrete colors),
        'temp' colors points by temperature values (continuous colormap),
        or None for single color.
    figsize : tuple
        Figure size for each subplot.
    save : bool
        If True, save figures to outdir instead of showing.
    outdir : str
        Directory for saving plots (created if necessary).
    alpha : float
        Marker alpha (transparency).
    s : int
        Marker size.
    """
    # basic checks
    required = ['GWM', 'DepthNN', 'logt']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"DataFrame must contain column '{c}'")

    os.makedirs(outdir, exist_ok=True)

    gwms = df['GWM'].unique()

    for gwm in gwms:
        group = df[df['GWM'] == gwm]
        if group.empty:
            continue

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Time vs Depth scatter for well {gwm}")
        ax.set_ylabel("Depth [m]")
        ax.invert_yaxis()  # depths increasing downward

        all_times = []
        all_depths = []
        all_c = []   # color values (serial or temp)
        all_labels = []  # for legend mapping

        # choose colormap / discrete colors if necessary
        if color_by == 'serial':
            uniques = group['Serial'].unique()
            cmap = plt.get_cmap('tab20')
            color_map = {ser: cmap(i % 20) for i, ser in enumerate(uniques)}
        elif color_by == 'temp':
            cmap = plt.get_cmap('viridis')
        else:
            cmap = None

        for _, row in group.iterrows():
            depth = row['DepthNN']
            times = row['logt']
            if times is None:
                continue

            # ensure times are datetime-like
            try:
                times_dt = pd.to_datetime(times)
            except Exception:
                # try mapping each element
                times_dt = pd.to_datetime([pd.to_datetime(t) for t in times])

            n = len(times_dt)
            if n == 0:
                continue

            depths_arr = np.full(n, depth)

            if color_by == 'serial':
                cols = [color_map[row['Serial']]] * n
                all_c.extend(cols)
            elif color_by == 'temp':
                # if logT present and list-like, use it; else nan
                if 'logT' in row and isinstance(row['logT'], list) and len(row['logT']) == n:
                    temps = np.array(row['logT'], dtype=float)
                else:
                    temps = np.full(n, np.nan)
                all_c.extend(temps)
            else:
                all_c.extend([None] * n)

            all_times.extend(times_dt)
            all_depths.extend(depths_arr)
            all_labels.append((row['Serial'], depth))  # for potential legend

        if len(all_times) == 0:
            plt.close(fig)
            continue

        # plotting
        times_np = pd.to_datetime(all_times)
        depths_np = np.array(all_depths)

        if color_by == 'serial':
            # discrete colored scatter
            colors = all_c
            ax.scatter(times_np, depths_np, c=colors, s=s, alpha=alpha)
            # create legend entries for serial numbers
            # pick one point per serial (the color_map)
            handles = []
            labels = []
            for ser, col in color_map.items():
                handles.append(plt.Line2D(
                    [0], [0], marker='o', color='w', markerfacecolor=col, markersize=6))
                labels.append(str(ser))
            ax.legend(handles, labels, title='Serial',
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        elif color_by == 'temp':
            temps = np.array(all_c, dtype=float)
            sc = ax.scatter(times_np, depths_np, c=temps,
                            cmap='viridis', s=s, alpha=alpha)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Temperature [°C]')
        else:
            ax.scatter(times_np, depths_np, color='C0', s=s, alpha=alpha)

        # formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        ax.set_xlabel("Time")

        if save:
            outfile = os.path.join(outdir, f"time_depth_scatter_{gwm}.png")
            plt.tight_layout()
            plt.savefig(outfile, dpi=200)
            plt.close(fig)
        else:
            plt.show()
