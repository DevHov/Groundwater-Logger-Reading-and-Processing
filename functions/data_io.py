# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:07:15 2025

@author: Hoffmann
"""
from pyrsktools import RSK


def read_logger_data(file_str):
    # RBR Handling
    if '.rsk' in file_str:
        with RSK(file_str) as rsk:
            raw = rsk.copy()
        raw.readdata()

        time = raw.data['timestamp']
        temp = raw.data['temperature']
        try:
            p = raw.data['pressure']
        except:
            p = False

        return time, temp, p

    # HOBO Handling
    if '.hobo' in file_str:
        print('hobo')
        # return time, temp, p
