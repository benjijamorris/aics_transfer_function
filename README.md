# AICS Transfer Function

[![Build Status](https://github.com/AllenCell/aics_transfer_function/workflows/Build%20Master/badge.svg)](https://github.com/AllenCell/aics_transfer_function/actions)
[![Documentation](https://github.com/AllenCell/aics_transfer_function/workflows/Documentation/badge.svg)](https://AllenCell.github.io/aics_transfer_function)
[![Code Coverage](https://codecov.io/gh/AllenCell/aics_transfer_function/branch/main/graph/badge.svg)](https://codecov.io/gh/AllenCell/aics_transfer_function)

Python package for building 3d computational transfer function models for light microscopy images

---


## Quick Start

To build a transfer function from a source domain (e.g., low resolution images) to a target domain (e.g., high resolution images), we assume the training data (pairs of images in both source and target domains) are prepared by the registration step (see: https://github.com/AllenCell/aics_tf_registration). After that, there are three main steps: (1) calculating intensity normalization parameters. (2) training , (3) testing

(1) intensity normalization parameter
```bash
python scripts/pre_process_calc.py --source_domain /path/to/source/domain/data --target_domain /path/to/target/domain/data
```

parameters will be printed out on command line screen.

(2) training

Take the intensity normalization parameters calculated in step 1 and prepare a training configuration file ([Example](https://github.com/AllenCell/aics_transfer_function/blob/main/aics_transfer_function/config/transfer_function_training_example.yaml)). Mostly, just updating the normalization parameters, data paths, model save path, experiment names, etc. Then

```bash
TF_run --config /path/to/training/config.yaml --model train
```

(3) testing

The testing step can be done in three different ways: 
* validation: both source domain images and target domain images (ground truth) are available. This can be used to validate the model performance. [Example configuration file](https://github.com/AllenCell/aics_transfer_function/blob/main/aics_transfer_function/config/transfer_function_validation_example.yaml) Then, `TF_run --config /path/to/validation/config.yaml --model validation`
* inference: only source domain images are available. This is used to make predictions on new data. [Exmaple configuration file](https://github.com/AllenCell/aics_transfer_function/blob/main/aics_transfer_function/config/transfer_function_inference_example.yaml). Then, `TF_run --config /path/to/inference/config.yaml --model inference`
* API for applying transfer function on numpy array: This is useful when you want to use transfer function as part of your big workflow. [Demo Jupyter Notebook](https://github.com/AllenCell/aics_transfer_function/blob/main/playbook/apply_tf_on_numpy_array.ipynb)

**Note about parameters in configuration yaml file**
In general, only the `datapath` and `normalization` sections need to change, no matter for training or testing. 
*`datapath`: the directory of source domain images and target domain images (no target when doing inference), as well as the directory to save prediction (only when testing)
*`normalization`: Only `simple_norm` using `middle_otsu` is supported currently. `simple_norm` refers intensity normalization by rescaling the intensity into `[m - a * s, m + b * s]`, where `m` and `s` are mean intensity and standard deviation of all "valid" voxels. Here, "valid" voxels are the middle chunk of the image stack, where the middle chunk is estimated by applying Otsu thresholding to roughly identify where the signals are. By doing this, we could reduce the impact of empty z-slices near the bottom and top of the stack. This is also where the name `middle_otsu` comes from.

Besides above two, there one section (called `save`) specific to training. Users can specify where to save the model (`results_folder`) and whether to save example predictions periodically (`save_training_inspections`), if so, how frequent to save (`print_freq`). Also, it is recommeded to set `save_latest_freq` (how frequent to save the model) the same as `print_freq`, so that the example prediction you observe matches the checkpoints being saved. 

One last important parameter is `path` under `load_trained_model` section. If this is used in training, it means the training will use this model as the initial model. If this is used in testing, it is the path to the model you want to apply. 

## Installation
**Stable Release:** `pip install aics_transfer_function`<br>
**Development Head:** `pip install git+https://github.com/AllenCell/aics_transfer_function.git`

## Documentation
For full package documentation please visit [AllenCell.github.io/aics_transfer_function](https://AllenCell.github.io/aics_transfer_function).

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.


The implementation of this repo was partially inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. A few core functions were reused. 

***Free software: Allen Institute Software License***

