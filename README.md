# README for Keystroke Hacking Detection Experiment

This repository contains sample code used in the article [Keystroke Hacking: Are We Still Safe?](https://medium.com/@kevinjesse/keystroke-hacking-are-we-still-safe-dc06d615c80d) by Kevin Jesse. The article discusses the feasibility of detecting keystroke hacking attempts through sound analysis, leveraging machine learning models to identify unique keystroke sounds. Below is an overview of the repository contents, including descriptions of the Python files and data sets used in this study. Many more experiments such as few shot train/eval and code used to generate the figures may be available upon request.

## Files Overview

### `train.py`

This is the main script used to train the machine learning model on the dataset of keystroke sounds. It utilizes PyTorch, a popular deep learning framework, and Accelerate by Hugging Face for simplifying distributed training. The script defines a custom dataset class (`MelDataset`), preprocessing functions for generating mel spectrograms from audio files (`make_melspec`), and functions for applying SpecAugment, a method for data augmentation (`SpecAugment`). The training loop includes the use of a CoAtNet model, defined in `coatnet.py`, for classification tasks.

### `coatnet.py`

Defines the CoAtNet model architecture, a convolutional and transformer network hybrid designed for efficient image classification tasks, which has been repurposed here for audio spectrogram classification. This file contains various helper functions and classes that construct the different layers and blocks of the CoAtNet model, including attention mechanisms, feed-forward networks, and squeeze-and-excitation blocks. Several versions of the CoAtNet model with varying depths and widths are available for use. The model code originated [here](https://github.com/chinhsuanwu/coatnet-pytorch/blob/master/coatnet.py).

## Data Files

### `data_webex.pkl` and `data_webex_holdout.pkl`

These two files are serialized pandas DataFrames containing the training and holdout datasets, respectively. Each DataFrame includes columns for waveform data of keystroke sounds and their corresponding labels. The `data_webex.pkl` file is used for training the model, while `data_webex_holdout.pkl` is used for evaluating its performance on unseen data. The datasets represent one of many potential experiments that can be conducted in the domain of keystroke sound analysis. Specifically, this experiment aims to demonstrate the challenges of generalizing models across different recording environments. The training and holdout datasets were collected under identical conditions other than the site of the recording (two different rooms in a home).

## Experiment Insights

The study highlights an important consideration in the field of keystroke detection through sound analysis: the generalizability of models across different recording environments. The results from training on `data_webex.pkl` and testing on `data_webex_holdout.pkl` reveal that models may struggle to perform well when exposed to data from different environments than they were trained on. This underscores the need for robust data collection and augmentation techniques to improve model resilience and reliability in real-world applications.
