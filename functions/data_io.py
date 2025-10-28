# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:07:15 2025

@author: Hoffmann
"""
import pandas as pd
from pyrsktools import RSK
import pyautogui
import time
import subprocess
import os
import glob
import re


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


def read_hobo(file_str, HOBOware_exe=r"C:\Program Files\Onset Computer Corporation\HOBOware\HOBOware.exe"):
    process = subprocess.Popen(HOBOware_exe)
    time.sleep(2)
    pyautogui.hotkey('ctrl', 'o')
    time.sleep(0.5)
    pyautogui.write(file_str)
    pyautogui.press('enter')
    time.sleep(0.5)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.hotkey('ctrl', 'e')
    time.sleep(0.5)
    pyautogui.hotkey('shift', 'tab')
    time.sleep(0.5)
    pyautogui.press('enter')
    pyautogui.write(file_str.replace('.hobo', 'autoconvert.csv'))
    pyautogui.press('enter')
    process.terminate()


def read_logger_data(file_str, HOBOware_exe):
    # Wrapper function handling logger data types.
    # RBR Handling
    if '.rsk' in file_str:
        return read_rbr(file_str)

    # HOBO Handling
    if os.name == 'nt':
        if '.hobo' in file_str:
            # Isolate serial number of current logfile
            serial = os.path.splitext(os.path.basename(file_str))[0][:8]

            # create a list of all existing csvs in the directory
            csv_path = os.path.join(os.path.dirname(file_str), '*.csv')
            csvs = glob.glob(csv_path)

            # check if the serial number of the current file exists
            csv = [val for val in csvs if serial in val]

            if len(csv) == 0:
                read_hobo(file_str)
                csv = file_str.replace('.hobo', 'autoconvert.csv')
            elif len(csv) > 1:
                print('There is more than one .csv file with serial number ',
                      serial, '. The last one is taken.')

            hobo_data = pd.read_csv(csv)

    elif os.name == 'posix':
        if '.csv' in file_str:
            return pd.read_csv(file_str)
        else:
            print('You\'re running your Code on a Unix system. Please ',
                  'export your files with HOBOware on windows first.')


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
