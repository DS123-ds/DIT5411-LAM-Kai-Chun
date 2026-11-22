# Baseline LeakyReLU CNN model checkpoint

Due to GitHub's 25MB upload limit for files uploaded via the web interface,  
the trained Keras model file `last.keras` is stored on Google Drive instead of this repository.

## Download link

You can download the final trained model used for reporting the assignment results from:

[https://your-google-drive-link-here](https://drive.google.com/drive/folders/1ouSN56sI4H9ztUxbw9rssTNtq11lVTT1?usp=sharing)

(This link is set to "Anyone with the link can view/download".)

## How to use

1. Download the file `last.keras` from the Google Drive link above.
2. Place it in the following path relative to this repository:

   `models/baseline_leaky_continuation/last.keras`

3. After that, the evaluation / inference code in the notebooks or `src/training_core.py`
   can load the model from this location.
