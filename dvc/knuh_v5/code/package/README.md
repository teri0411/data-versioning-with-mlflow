# CovSF

## Environment
We have described all the required package information in the environment.yml file. This file ensures that all necessary dependencies are installed correctly.

If you are using anaconda, 

    conda env create -f environment.yml

this code will create 'covsf' environment.

## Usage
1. If you are using Anaconda, activate the environment where all required packages are installed.
    `conda activate covsf`
2. Running type
    - Run CovSF 
                `python covsf.py -r [INPUT_FILE] [SAVE_DIR]`
        - Input file format : The input file format is the same as described on the web manual page
            or use files in example/ as reference.
        - SAVE_DIR : if None, result csv will be saved on current dir.
        - Output file : result.csv
            Each row’s +0, +1, +2, +3 indicate the output of CovSF when the input sequence’s last day is the current row
            CovSF scores are calculated based on the model’s outputs across all input days.
        - Example
            `python covsf.py -r example/RP_1.csv`
