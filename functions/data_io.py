# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:07:15 2025

@author: DevHov
"""
import pandas as pd
from pyrsktools import RSK
import pyautogui
import time
import subprocess
import os
import user_definitions as ud


def lang(lang):
    if lang == 'en':
        return '(?i)temp', '(?i)press'
    if lang == 'de':
        return '(?i)temp', '(?i)druck'


def read_rbr(file_str):
    try:
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
    time = pd.to_datetime(hobo_data_raw.iloc[:, 1], errors='coerce').to_list()

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
    print('Starting process for: ', os.path.basename(file_str))
    # Wrapper function handling logger data types.
    # RBR Handling
    if '.rsk' in file_str:
        return read_rbr(file_str)

    # HOBO Handling
    file_str_csv = file_str.replace('.hobo', '.csv')
    if os.name == 'nt':
        # if matchin csv does not exist yet create one
        if '.hobo' in file_str and not os.path.exists(file_str_csv):
            hobo_to_csv(file_str, HOBOware_exe)
            try:
                size = os.path.getsize(file_str_csv)
                delay = 1
                msg = 'Empty file created. Retrying with ', delay, 'sec delay...'
            except:
                print('Error during HOBO export, retrying with 1 sec delay...')
                hobo_to_csv(file_str, 1, HOBOware_exe)
                size = os.path.getsize(file_str_csv)
                delay = 2
                msg = 'Another error occured. Retrying again with ', delay, 'sec delay...'

            if size == 0:
                print(msg)
                hobo_to_csv(file_str, delay, HOBOware_exe)

    elif os.name == 'posix':
        if not os.path.exists(file_str_csv):
            errormessage = ('You\'re running your Code on a Unix system. Please',
                            ' export your files with HOBOware on windows first.')
            raise ValueError(errormessage)

    if os.path.exists(file_str_csv):
        return read_hobo_csv(file_str)
    else:
        raise ValueError('No matching file found')


def subtract_reference(df, reference_serial, tolerance_hours=12,
                       sercol='Serial', tcol='logt', Tcol='logT', pcol='logp',
                       NNcol='DepthNN', namcol='GWM', mancol='Abstich'):
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
                'GWL_NN': merged['GWL_NN'].tolist(),
                'man_calib': merged['Abstich'].tolist()
            })

    return pd.DataFrame(results)
