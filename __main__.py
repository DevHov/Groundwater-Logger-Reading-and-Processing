# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:16:07 2025

@author: Hoffmann
"""

import os
import glob
from pyrsktools import RSK

filepath_loggerdata = os.path.join(os.getcwd(), 'Loggerdata', '*.rsk')
files = glob.glob(filepath_loggerdata)


# %%

with RSK(files[7]) as rsk:
    raw = rsk.copy()
    raw.readdata()
