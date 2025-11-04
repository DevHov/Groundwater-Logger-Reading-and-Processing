# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:16:07 2025

@author: DevHov

"""

import matplotlib.dates as mdates
import numpy as np
import os
import glob
import pandas as pd
import user_definitions as ud
import functions.data_io as fdo
import functions.plotting as fpl
import matplotlib.pyplot as plt

subfolder_loggerdata = 'Loggerdata'
root_folder = os.path.dirname(os.getcwd())
filepath_rsk = os.path.join(root_folder, subfolder_loggerdata, '*.rsk')
filepath_hobo = os.path.join(root_folder, subfolder_loggerdata, '*.hobo')
filepath_xlsx = os.path.join(root_folder, subfolder_loggerdata, '*.xlsx')

files_rsk = glob.glob(filepath_rsk)
files_hobo = glob.glob(filepath_hobo)
files_logger = files_rsk + files_hobo
files_xlsx = glob.glob(filepath_xlsx)
# ignore excel meta files
files_xlsx = [x for x in files_xlsx if '~$' not in x]

if len(files_xlsx) > 1:
    raise ValueError('More than one logger-definition-file.')
elif len(files_xlsx) == 0:
    raise ValueError('No logger-definition-file found.')
# %%
ldf = pd.read_excel(files_xlsx[0])

out = pd.DataFrame()
for i, snr in enumerate(ldf.Seriennummer):
    match = [s for s in files_logger if str(snr) in s]
    if len(match) > 1:
        print('WARNING: More than one Logfile for serial number ' + str(snr))
        print('Firs dataset is used.')
    elif len(match) == 0:
        print('WARNING: No Logfile found for serial number ' + str(snr))
        print('Logger skipped.')
        continue

    match = match[0]
    [time, temp, p, logger_type] = fdo.read_logger_data(match, ud.HOBOware_exe)

    out_i = {'Serial': snr,
             'type': logger_type,
             'GWM': ldf['GWM'][i],
             'DepthNN': ldf['Tiefe NN'][i],
             'GWL_NN': ldf['GWL NN'][i],
             'logt': time,
             'logT': temp,
             'logp': p}
    out = pd.concat([out, pd.DataFrame([out_i])], ignore_index=True)


# %% Corrected Pressure logger data:
serial_number_reference = ud.barometric_reference_logger
groundwater_level = fdo.subtract_reference(out, serial_number_reference)

fig, ax = plt.subplots()
fpl.plot_gw(ax, groundwater_level, 'time',
            'GWL_NN', 'Groundwater Level $(mNN)$')
plt.show()

fig, ax = plt.subplots()
fpl.plot_gw(ax, groundwater_level, 'time', 'temp',
            'Groundwater Temperature $(°C)$')
plt.show()

# %%
# fig, ax = plt.subplots()
for gwm, group in out.groupby('GWM'):
    # Ignore if Reference
    if group['Serial'].iloc[0] == ud.barometric_reference_logger:
        continue
    else:
        print(f"Plot für Messstelle {gwm}")

    # 3 Für jeden Logger (Tiefe) Daten extrahieren
    fig, ax = plt.subplots(figsize=(8, 6))

    # Leere Listen für Zeit, Tiefe, Temperatur
    all_times = []
    all_depths = []
    all_temps = []

    for _, row in group.iterrows():
        times = np.array(row['logt'])
        temps = np.array(row['logT'])
        depth = row['DepthNN']

        # 🔹 NaN-Werte entfernen
        valid = ~np.isnan(temps)
        times = times[valid]
        temps = temps[valid]

        # Tiefe auf gleiche Länge wie Zeitreihe bringen
        depths = np.full_like(temps, depth, dtype=float)

        all_times.append(times)
        all_temps.append(temps)
        all_depths.append(depths)

    # Alle in Arrays zusammenführen
    all_times = np.concatenate(all_times)
    all_temps = np.concatenate(all_temps)
    all_depths = np.concatenate(all_depths)

    # 4️⃣ In 2D-Gitter bringen: Zeit (x), Tiefe (y), Temperatur (Farbwert)
    # Dazu sortieren und Meshgrid bauen
    sort_idx = np.argsort(all_times)
    all_times = all_times[sort_idx]
    all_temps = all_temps[sort_idx]
    all_depths = all_depths[sort_idx]

    # Optional: Interpolieren auf gleichmäßiges Raster
    # Beispiel: wöchentliche Intervalle, Tiefenraster alle 0.5 m
    time_unique = pd.to_datetime(pd.Series(all_times)).sort_values().unique()
    depth_unique = np.sort(group['DepthNN'].unique())

    # Temperaturmatrix erzeugen (T [Tiefe, Zeit])
    temp_matrix = np.full((len(depth_unique), len(time_unique)), np.nan)

    for i, depth in enumerate(depth_unique):
        sel = group[group['DepthNN'] == depth].iloc[0]

        # Zeitreihe als DatetimeIndex
        t = pd.to_datetime(sel['logt'])
        T = pd.Series(sel['logT'], index=t).astype(float)

        # 🔹 NaN-Werte raus
        T = T.dropna()

        # 🔹 Zeitreihe sortieren (Fehlerquelle!)
        T = T.sort_index()

        if T.empty:
            continue  # Falls nach dem Filtern keine Werte mehr übrig sind

        # 🔹 Jetzt darf reindex mit 'nearest' verwendet werden
        temp_series = T.reindex(time_unique, method='nearest')

        temp_matrix[i, :] = temp_series.values

    # 5️⃣ Plotten als Colormap
    X, Y = np.meshgrid(time_unique, depth_unique)

    pcm = ax.pcolormesh(X, Y, temp_matrix, shading='auto', cmap='turbo')
    ax.invert_yaxis()  # Tiefer = unten
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.colorbar(pcm, label='Temperatur [°C]')
    ax.set_xlabel('Zeit')
    ax.set_ylabel('Tiefe [m]')
    ax.set_title(f"Temperaturverlauf in Messstelle {gwm}")

    plt.tight_layout()
    plt.show()
