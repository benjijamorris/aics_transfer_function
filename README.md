# AICS Transfer Function

[![Build Status](https://github.com/AllenCell/aics_transfer_function/workflows/Build%20Master/badge.svg)](https://github.com/AllenCell/aics_transfer_function/actions)
[![Documentation](https://github.com/AllenCell/aics_transfer_function/workflows/Documentation/badge.svg)](https://AllenCell.github.io/aics_transfer_function)
[![Code Coverage](https://codecov.io/gh/AllenCell/aics_transfer_function/branch/main/graph/badge.svg)](https://codecov.io/gh/AllenCell/aics_transfer_function)

Python package for building 3d computational transfer function models for light microscopy images

---


## Quick Start

To build a transfer function from a source domain (e.g., low resolution images) to a target domain (e.g., high resolution images), we assume the training data (pairs of images in both source and target domains) are prepared by the registration step (see: https://github.com/AllenCell/aics_tf_registration). After that, there are 3 main steps: (1) calculating intensity normalization parameters. (2) training , (3) testing

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

## Installation
**Stable Release:** `pip install aics_transfer_function`<br>
**Development Head:** `pip install git+https://github.com/AllenCell/aics_transfer_function.git`

## Documentation
For full package documentation please visit [AllenCell.github.io/aics_transfer_function](https://AllenCell.github.io/aics_transfer_function).

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.


The implementation of this repo was partially inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix. A few core functions were reused. 

***Free software: Allen Institute Software License***

