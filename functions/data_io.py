# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:07:15 2025

@author: DevHov
"""
import logging
import glob
import pandas as pd
from pyrsktools import RSK
import pyautogui
import time
import subprocess
import os
import user_definitions as ud
import numpy as np

# %% General Functions


def lang(lang):
    if lang == 'en':
        return '(?i)temp', '(?i)press'
    if lang == 'de':
        return '(?i)temp', '(?i)druck'


def log_message(msg, flair='info'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    flair = flair.lower()
    if flair == "info":
        logger.info(msg)
    elif flair == "warning":
        logger.warning(msg)
    elif flair == "error":
        logger.error(msg)
    elif flair == "debug":
        logger.debug(msg)
    elif flair == "critical":
        logger.critical(msg)
    else:
        logger.info(f"(unknown flair '{flair}') {msg}")

# %% Import Raw Logger Information


def load_logger_definition(logger_dirname, root_dir=None):
    if not root_dir:
        root_dir = os.path.dirname(os.getcwd())
    filepath_xlsx = os.path.join(root_dir, logger_dirname, '*.xlsx')
    files_xlsx = glob.glob(filepath_xlsx)
    # ignore excel meta files
    files_xlsx = [x for x in files_xlsx if '~$' not in x]

    if len(files_xlsx) > 1:
        raise ValueError('More than one logger-definition-file.')
    elif len(files_xlsx) == 0:
        raise ValueError('No logger-definition-file found.')

    return pd.read_excel(files_xlsx[0])


def load_logger_data(logger_dirname, root_dir=None):
    if not root_dir:
        root_dir = os.path.dirname(os.getcwd())
    filepath_rsk = os.path.join(root_dir, logger_dirname, '*.rsk')
    filepath_hobo = os.path.join(root_dir, logger_dirname, '*.hobo')

    files_rsk = glob.glob(filepath_rsk)
    files_hobo = glob.glob(filepath_hobo)
    return files_rsk + files_hobo


def find_matching_logfile(snr, files_logger, logger=None):
    match = [s for s in files_logger if str(snr) in s]
    if len(match) > 1:
        msg = f'More than one Logfile for serial number {snr}. Firs dataset is used.'
        log_message(msg, 'warning')
    elif len(match) == 0:
        msg = f'No Logfile found for serial number {snr}. Logger skipped.'
        log_message(msg, 'warning')
        return None
    return match[0]

# Import logger Data


def read_rbr(file_str):
    try:
        with RSK(file_str) as rsk:
            raw = rsk.copy()
        raw.readdata()

        time = raw.data['timestamp'].tolist()
        temp = raw.data['temperature'].tolist()
        try:
            p = raw.data['pressure'].tolist()
        except:
            p = -999

        return time, temp, p, raw.instrument.model
    except:
        print('Error while handling rbr file ', file_str)


def hobo_to_csv(file_str,
                sleeptime=0.5,
                HOBOware_exe=r"C:\Program Files\Onset Computer Corporation\HOBOware\HOBOware.exe"):
    process = subprocess.Popen(HOBOware_exe)
    # waiting until program is open
    while True:
        if 'HOBOware' in pyautogui.getAllTitles():
            break
        elif 'HOBOware Pro' in pyautogui.getAllTitles():
            break
        time.sleep(0.1)
    pyautogui.getWindowsWithTitle('HOBOware')[0].activate()
    pyautogui.getWindowsWithTitle('HOBOware Pro')[0].activate()
    pyautogui.hotkey('ctrl', 'o')
    time.sleep(sleeptime)
    pyautogui.write(file_str)
    pyautogui.press('enter')
    time.sleep(sleeptime)
    pyautogui.press('enter')
    time.sleep(sleeptime)
    pyautogui.hotkey('ctrl', 'e')
    time.sleep(sleeptime)
    pyautogui.hotkey('shift', 'tab')
    time.sleep(sleeptime)
    pyautogui.press('enter')
    time.sleep(sleeptime)
    # pyautogui.write(file_str.replace('.hobo', 'autoconvert.csv'))
    pyautogui.press('enter')
    process.terminate()

    print('Successfully converted: ', os.path.basename(file_str))


def read_hobo_csv(file_str):
    file_str_csv = file_str.replace('.hobo', '.csv')

    hobo_data_raw = pd.read_csv(file_str_csv, skiprows=[0])
    time = pd.to_datetime(
        hobo_data_raw.iloc[:, 1], format='%d.%m.%y %I:%M:%S %p', errors='coerce').to_list()

    try:
        temp_raw = hobo_data_raw.filter(regex=lang(ud.lang)[0])
        temp = temp_raw.iloc[:, 0].to_list()
    except:
        temp = [None] * len(hobo_data_raw)
    try:
        pressure_raw = hobo_data_raw.filter(regex=lang(ud.lang)[1])
        p = pressure_raw.iloc[:, 0].to_list()
    except:
        p = -999

    return time, temp, p, 'HOBO'


def read_logger_data(file_str, HOBOware_exe):
    msg = f'Starting process for: {os.path.basename(file_str)}'
    log_message(msg)
    # Wrapper function handling logger data types.
    # RBR Handling
    if '.rsk' in file_str:
        rbr_data = read_rbr(file_str)
        log_message(f'{os.path.basename(file_str)} loaded.')
        return rbr_data

    # HOBO Handling
    file_str_csv = file_str.replace('.hobo', '.csv')
    if os.name == 'nt':
        # if matchin csv does not exist yet create one
        if '.hobo' in file_str and not os.path.exists(file_str_csv):
            log_message(f'No csv found for {file_str}')
            hobo_to_csv(file_str, HOBOware_exe=HOBOware_exe)
            try:
                size = os.path.getsize(file_str_csv)
                delay = 1
                msg = 'Empty file created. Retrying with ', delay, 'sec delay...'
            except:
                msg = 'Error during HOBO export, retrying with 1 sec delay...'
                log_message(msg, 'warning')
                hobo_to_csv(file_str, 1, HOBOware_exe)
                size = os.path.getsize(file_str_csv)
                delay = 2
                msg = 'Another error occured. Retrying again with ', delay, 'sec delay...'

            if size == 0:
                log_message(msg, 'warning')
                hobo_to_csv(file_str, delay, HOBOware_exe)

    elif os.name == 'posix':
        if not os.path.exists(file_str_csv):
            errormessage = ('You\'re running your Code on a Unix system. Please',
                            ' export your files with HOBOware on windows first.')
            raise ValueError(errormessage)

    if os.path.exists(file_str_csv):
        hobo_csv = read_hobo_csv(file_str)
        log_message(f'{os.path.basename(file_str_csv)} loaded.')
        return hobo_csv
    else:
        raise ValueError('No matching file found')

# %% Data Refinement


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
