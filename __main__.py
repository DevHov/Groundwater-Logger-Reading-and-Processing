# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:16:07 2025

@author: DevHov

"""

import pandas as pd
import user_definitions as ud
import functions.data_io as fdo
import functions.plotting as fpl
import matplotlib.pyplot as plt


files_logger = fdo.load_logger_data(ud.logger_dirname)
logger_definition = fdo.load_logger_definition(ud.logger_dirname)
# %%


data_raw = pd.DataFrame()
for i, snr in enumerate(logger_definition['Seriennummer']):
    # Filter Loggerst
    match = fdo.find_matching_logfile(snr, files_logger)
    if match is None:
        continue

    if fdo.last_HOBOware_call(match, files_logger):
        terminate_HOBOware = ud.terminate_HOBOware
    else:
        terminate_HOBOware = False

    [time, temp, p, logger_type] = fdo.read_logger_data(
        match, ud.HOBOware_exe, terminate_HOBOware)

    data_i = {'Serial': snr,
              'type': logger_type,
              'GWM': logger_definition['GWM'][i],
              'DepthNN': logger_definition['Tiefe NN'][i],
              'GWL_NN': logger_definition['GWL NN'][i],
              'logt': time,
              'logT': temp,
              'logp': p}
    data_raw = pd.concat([data_raw, pd.DataFrame([data_i])], ignore_index=True)


# %% Corrected Pressure logger data:
serial_number_reference = ud.barometric_reference_logger
data = fdo.clean_dataset(data_raw, serial_number_reference)

# %%
fig, ax = plt.subplots()
fpl.plot_gw(ax, data, 'logt', 'log_gwl', 'Groundwater Level $(mNN)$')
plt.show()
# %%
fig, ax = plt.subplots()
fpl.plot_gw(ax, data, 'logt', 'logT', 'Groundwater Temperature $(°C)$')
plt.show()


# %% Temperature depth Plot

fpl.plot_temperature_profiles(data)

# %%
fpl.plot_time_depth_scatter(data)
