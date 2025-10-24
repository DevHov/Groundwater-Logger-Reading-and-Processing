# Groundwater-Processing

This code is still in very early development before version alpha.



This code repository contains a general toolset and workflow for reading and
processing files from logger chains of RBR solo and RBR duet as well as Onset
HOBO U20L and HOBO Water Temp Pro v2 (pending). Scope is to create a general
workflow containing the following steps:



1. raw data import of \*.rsk and \*.hobo import
2. barometric compensation with reference logger
3. time-series plot for each well in terms of:
   3.1 groundwater level
   3.2 temperature distribution as 2d heatmap (x: time, y: depth,
   color: temperature)
   3.3 combined plot

The code development is still in very early stages.

# folder structure

```

root\_folder
├── Repository          spyder project folder, cwd)
|   ├── **main**.py
|   └── functions
|       └── data\_io.py  input/output functions
|
├── Loggerdata
|   ├── \*.rsk           RBR Logger Data (naming see below)
|   ├── \*.csv          Hobo Logger Data (naming see below)
|   └── \*.xlsx          Metadata and Installation Data (naming see below)
|
└── Results            Is created by the code
├── Data           Contains all generated data
└── Plots
└── \*.png      Plotted graphs

```

# further naming conventions

*.rsk naming scheme (default)
xxxxxx\_%Y%m%d\_%H%M*.rsk
xxxxxx 6-digit serial number of the logger
\* further commentary possible e.g. well name

*.csv naming scheme (default)
xxxxxxxx*.csv
xxxxxxxx 8-digit serial number of the logger
\* further commentary possible e.g. well name

\*.xlsx columns:
GWM - Name of the Well
Tiefe NN - Depth of the Logger (NN)
Seriennummer - Logger Serial Number

