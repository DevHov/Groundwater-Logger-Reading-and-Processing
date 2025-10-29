# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:16:07 2025

@author: DevHov

"""

import os
import glob
import pandas as pd
import user_definitions as ud
import functions.data_io as fdo
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
             'logt': time,
             'logT': temp,
             'logp': p}
    out = pd.concat([out, pd.DataFrame([out_i])], ignore_index=True)


# %% Corrected Pressure logger data:
serial_number_reference = ud.barometric_reference_logger
groundwater_level = fdo.subtract_reference(out, serial_number_reference)

fig, ax = plt.subplots()
for _, row in groundwater_level.iterrows():
    ax.plot(row['time'], row['GWL_NN'], label=row['name'])
ax.set_xlabel('Date')
ax.set_ylabel('Groundwater Level $(mNN)$')
ax.legend()


fig, ax = plt.subplots()
for _, row in groundwater_level.iterrows():
    ax.plot(row['time'], row['temp'], label=row['name'])
ax.set_xlabel('Date')
ax.set_ylabel('Groundwater Temperature $(°C)$')
ax.legend()
