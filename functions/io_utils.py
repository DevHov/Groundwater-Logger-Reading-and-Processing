# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:07:15 2025

@author: DevHov

IO utilities for reading HOBO and RBR logger data, managing file exports,
and communicating with HOBOware.
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
        root_dir = ud.working_directory
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
        root_dir = ud.working_directory
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
        log_message(f'Error while handling rbr file {file_str}.', 'error')


def hobo_started():
    """Checks if HOBOware or HOBOware pro is in open tabs."""
    if 'HOBOware' in pyautogui.getAllTitles() or 'HOBOware Pro' in pyautogui.getAllTitles():
        return True
    else:
        return False


def hobo_to_csv(file_str,
                sleeptime=0.5,
                terminate_hoboware=True,
                HOBOware_exe=r"C:\Program Files\Onset Computer Corporation\HOBOware\HOBOware.exe"):

    # Starts hoboware if neccessary
    if not hobo_started():
        subprocess.Popen(HOBOware_exe)
        log_message('Starting HOBOware software.', 'info')
        # waiting until program is open
        while True:
            if hobo_started():
                break
            time.sleep(0.1)

    # Move software to cursor focus
    pyautogui.getWindowsWithTitle('HOBOware')[0].activate()
    pyautogui.getWindowsWithTitle('HOBOware Pro')[0].activate()

    # Open hobo file
    pyautogui.hotkey('ctrl', 'o')
    time.sleep(sleeptime)
    pyautogui.write(file_str)
    pyautogui.press('enter')
    time.sleep(sleeptime)
    pyautogui.press('enter')
    time.sleep(sleeptime)

    # Export to csv file
    pyautogui.hotkey('ctrl', 'e')
    time.sleep(sleeptime)
    pyautogui.hotkey('shift', 'tab')
    time.sleep(sleeptime)
    pyautogui.press('enter')
    time.sleep(sleeptime)
    # pyautogui.write(file_str.replace('.hobo', 'autoconvert.csv'))
    pyautogui.press('enter')
    time.sleep(sleeptime)
    if terminate_hoboware:
        time.sleep(sleeptime)
        log_message('Stopping HOBOware software.', 'info')
        pyautogui.hotkey('alt', 'f4')

    log_message(
        f'Successfully converted: {os.path.basename(file_str)}.', 'info')


def last_HOBOware_call(current_file, file_list):
    """
    Checks whether the current file is the *last* HOBO file in the list
    that does NOT yet have a matching .csv export.

    Parameters
    ----------
    current_file : str
        Path to the current logger file being processed.
    files_logger : list of str
        List of all logger file paths in the folder.

    Returns
    -------
    bool
        True if current_file is the last unexported HOBO file,
        False otherwise.
    """
    # List all HOBO files
    hobo_files = [f for f in file_list if f.lower().endswith('.hobo')]

    # List all HOBO files with no corresponding csv file
    unexported = [f for f in hobo_files if not os.path.exists(
        f.replace('.hobo', '.csv'))]

    # return false if all HOBO files are exported
    if not unexported:
        return False

    # last unexported file
    last_unexported = unexported[-1]

    # check if current file is last file
    return os.path.abspath(current_file) == os.path.abspath(last_unexported)


def read_hobo_csv(file_str):
    file_str_csv = file_str.replace('.hobo', '.csv')

    hobo_data_raw = pd.read_csv(file_str_csv, skiprows=[0])
    time = pd.to_datetime(
        hobo_data_raw.iloc[:, 1], errors='coerce').to_list()

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


def read_logger_data(file_str, HOBOware_exe, terminate_hoboware=True):
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
            hobo_to_csv(file_str, HOBOware_exe=HOBOware_exe,
                        terminate_hoboware=terminate_hoboware)
            try:
                size = os.path.getsize(file_str_csv)
                delay = 1
                msg = 'Empty file created. Retrying with ', delay, 'sec delay...'
            except:
                msg = 'Error during HOBO export, retrying with 1 sec delay...'
                log_message(msg, 'warning')
                hobo_to_csv(file_str, 1, terminate_hoboware, HOBOware_exe)
                size = os.path.getsize(file_str_csv)
                delay = 2
                msg = 'Another error occured. Retrying again with ', delay, 'sec delay...'

            if size == 0:
                log_message(msg, 'warning')
                hobo_to_csv(file_str, delay, terminate_hoboware, HOBOware_exe)

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
