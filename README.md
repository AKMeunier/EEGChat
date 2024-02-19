# EEGChat

## Publication
This repository contains code related to the paper "A Conversational Brain-Artificial Intelligence Interface" by Anja Meunier, Michal Robert Zák, Lucas Munz,
Sofiya Garkot, Manuel Eder, Jiachen Xu, and Moritz Grosse-Wentrup.

## Setting up
Create a new environment (e.g. with conda) and install the requirements. Please use python 3.8 (for compatibility with psychopy).
```
conda create -n bai python=3.8
conda activate bai
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
conda install -c conda-forge liblsl
```
To reproduce the analysis of the paper, download the experiment data here: https://ucloud.univie.ac.at/index.php/s/ojeJ36pdKFrDBcj
Unzip it and put it into a top-level folder called "data".


## Running an experiment
To run EEGChat, first start decoder_main.py, and when it says "Waiting for experiment_start trigger..." start the experiment_main.py script. 

