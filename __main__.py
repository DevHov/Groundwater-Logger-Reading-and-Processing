# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:16:07 2025

@author: Hoffmann

"""

import os
import glob
import pandas as pd
import user_definitions as ud
import functions.data_io as fdo

subfolder_loggerdata = 'Loggerdata'
root_folder = os.path.dirname(os.getcwd())
filepath_rsk = os.path.join(root_folder, subfolder_loggerdata, '*.rsk')
filepath_hobo = os.path.join(root_folder, subfolder_loggerdata, '*.hobo')
filepath_xlsx = os.path.join(root_folder, subfolder_loggerdata, '*.xlsx')

files_rsk = glob.glob(filepath_rsk)
files_hobo = glob.glob(filepath_hobo)
files_logger = files_rsk + files_hobo
files_xlsx = glob.glob(filepath_xlsx)

if len(files_xlsx) > 1:
    raise ValueError('More than one logger-definition-file.')
elif len(files_xlsx) == 0:
    raise ValueError('No logger-definition-file found.')
# %%

# loading datasets
# loading definitions
ldf = pd.read_excel(files_xlsx[0])

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
    ds = fdo.read_logger_data(match)


# %% Work in progress
snr_ref = ud.barometric_reference_logger
''' Pseudocode
ref_p = ds[snr_ref].p
for i, dsi in enumerate(ds):
    if dsi[snr] != snr_p:
        dsi.gwd = dsi.p - ref_p + dsi.znn
        '''
