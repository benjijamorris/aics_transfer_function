#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from .base_model import BaseModel


def find_model_using_name(model_name):
    model_filename = "aics_transfer_function.models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise ValueError(f"model {model_name} cannot be found")

    return model


def create_model(options):
    """
    Create a model given the option.

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(options.model)
    return model(options)
