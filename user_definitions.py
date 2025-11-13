# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 16:33:31 2025

@author: DevHov
"""
import os

# Serial Number of barometric reference logger
barometric_reference_logger = 208750

lang = 'de'  # Language of Hobo files

# State if plots are saved or just shown in ipython console
save_plots = True

working_directory = os.path.dirname(os.getcwd())

# Set true if issues the hobo files arise
hobo_debug = False

#############################################
# Optional:
logger_dirname = 'Loggerdata'

# Terminate the Hoboware Software after the last Conversion
terminate_HOBOware = True

# Define another path for the HOBOware program.
HOBOware_exe = r"C:\Program Files\Onset Computer Corporation\HOBOware\HOBOware.exe"
