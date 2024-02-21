# About 
The following folders contain the code and data used for the full-sentence generation part of EEGChat. As the full-sentence generator is a fine-tuned version of GPT-3 we cannot directly provide it. This repoistory contains all necessery code and data to create your own fine-tuned verison.

## Note
Since this code was written, OpenaAI has updated their fine-tuning API hence the dataset used for fine-tuning and the code used for fine-tuning need to be adapted first.

# Installation
Either use the conda environmnt from the global project scope or create a python environmnet and install the packages in `requirements.txt`.

```
pip install -r requirements.txt
```

If you run into any trouble when running the code, the code is confirmed working on `python 3.8.18` using the package versions inside `requirements-frozen.txt`.

# Structure

This repository contains two folders:
 1. Evaluation: Contains all code and data used when evaluating the created fine tunes
 2. Fine-Tuning: Contains all code and data necessery to create a fine-tuned full-sentence generator 

