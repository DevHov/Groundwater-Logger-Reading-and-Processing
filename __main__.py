# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:16:07 2025

@author: DevHov

Main execution script for groundwater logger data processing.

This script:
1. Loads raw logger and metadata files.
2. Converts HOBO/RBR data into pandas DataFrames.
3. Cleans and synchronizes datasets relative to a barometric reference logger.
4. Generates groundwater level and temperature plots.
"""

import os
import pandas as pd
import user_definitions as ud
import functions.io_utils as fio
from functions.data_processing import clean_dataset
import functions.plotting as fpl
import matplotlib.pyplot as plt


def main():
    files_logger = fio.load_logger_data(ud.logger_dirname)
    logger_definition = fio.load_logger_definition(ud.logger_dirname)

    data_raw = pd.DataFrame()
    records = []
    for i, serial_number in enumerate(logger_definition['Seriennummer']):
        fio.log_message(
            f"Processing {serial_number} ({i+1}/{len(logger_definition)})")
        # Filter Loggerst
        logger_file = fio.find_matching_logfile(serial_number, files_logger)
        if logger_file is None:
            continue

        if fio.last_HOBOware_call(logger_file, files_logger):
            terminate_HOBOware = ud.terminate_HOBOware
        else:
            terminate_HOBOware = False

        [time, temp, p, logger_type] = fio.read_logger_data(
            logger_file, ud.HOBOware_exe, terminate_HOBOware)

        logger_entry = {'Serial': serial_number,
                        'type': logger_type,
                        'GWM': logger_definition['GWM'][i],
                        'DepthNN': logger_definition['Tiefe NN'][i],
                        'GWL_NN': logger_definition['GWL NN'][i],
                        'logt': time,
                        'logT': temp,
                        'logp': p}
        records.append(logger_entry)

    data_raw = pd.DataFrame(records)

    # Corrected Pressure logger data:
    serial_number_reference = ud.barometric_reference_logger
    data = clean_dataset(data_raw, serial_number_reference)

    fig, ax = plt.subplots()
    fpl.plot_gw(ax, data, 'logt', 'log_gwl', 'Groundwater Level $(mNN)$')

    if ud.save_plots:
        plotdir = os.path.join(ud.working_directory, 'plots')
        os.makedirs(plotdir, exist_ok=True)
        outfile = os.path.join(plotdir, "hydrographic_curve.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=200)
        fio.log_message(f"Saved plot to {outfile}.", 'info')
        plt.close(fig)
    else:
        plt.show()

    # Temperature depth Plot
    fpl.plot_temperature_profiles(data, save=ud.save_plots, outdir=plotdir)

    if ud.hobo_debug:
        fpl.plot_time_depth_scatter(data, save=ud.save_plots, outdir=plotdir)


if __name__ == "__main__":
    main()
