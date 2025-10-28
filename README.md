# Groundwater-Processing

This code is still in very early development before version alpha.



This code repository contains a general toolset and workflow for reading and
processing files from logger chains of RBR solo and RBR duet as well as Onset
HOBO U20L and HOBO Water Temp Pro v2 (pending). Scope is to create a general
workflow containing the following steps:



1. raw data import of \*.rsk and \*.hobo or \.csv import
    1.1 for \*.rsk the [pyRSKtools](https://docs-static.rbr-global.com/pyrsktools/index.html) from RBR is used
    1.2 for \*.hobo due to the encryption of the files the official HOBOware
    Software (Windows) must be used. The GUI is called by the script and
    automatically saves a converted csv in the same folder. This process is
    only called once. If there is already a csv file with amatching serial
    number the csv is directly imported (this can be done in any OS)
    1.3 for \*.csv for any other type of logger data Some preprocessing might
    be needed - see naming conventions.
2. barometric compensation with reference logger
3. Calibration with initial manual measurement
4. time-series plot for each well in terms of:
   4.1 groundwater level
   4.2 temperature distribution as 2d heatmap (x: time, y: depth,
   color: temperature)
   4.3 combined plot

The code development is still in very early stages.

# folder structure

```

root\_folder
├── Repository          spyder project folder, cwd
|   ├── __main__.py
|   └── functions
|       └── data_io.py  input/output functions
|
├── Loggerdata
|   ├── *.rsk           RBR Logger Data (naming see below)
|   ├── *.csv          Hobo Logger Data (naming see below)
|   └── *.xlsx          Metadata and Installation Data (naming see below)
|
└── Results            Is created by the code
├── Data           Contains all generated data
└── Plots
└── *.png      Plotted graphs

```

# further naming conventions

### \*.rsk naming scheme (default)
- xxxxxx\_%Y%m%d\_%H%M\*.rsk
- xxxxxx 6-digit serial number of the logger
- \* further commentary possible e.g. well name

### \*.csv naming scheme (default)
- xxxxxxxx\*.csv
- xxxxxxxx 8-digit serial number of the logger
- \* further commentary possible e.g. well name

### \*.xlsx columns:
- GWM - Name of the Well
- Tiefe NN - Depth of the Logger (NN)
- Seriennummer - Logger Serial Number
- GWL NN - Groundwater level at measurement start for calibration (NN)

### \.csv columns:
- Date (datetime)
- Temperature (float)
- Pressure (float) - optional

