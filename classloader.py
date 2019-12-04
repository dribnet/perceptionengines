import argparse
import sys
import math
import importlib
from PIL import Image, ImageDraw
import numpy as np

class ScoringInterface:
    def predict(self, batch, explain=False):
        pass

    def get_target_size(self, batch):
        return None

    def get_input_preprocessor(self, batch):
        return None


def helpful_interface_message_exit(model_interface, e):
    print("==> Failed to load supporting class {}".format(model_interface))
    print("==> Check that package {} is installed".format(model_interface.split(".")[0]))
    print("(exception was: {})".format(e))
    sys.exit(1)

def load_image_function(fullname):
    model_class_parts = fullname.split(".")
    model_class_name = model_class_parts[-1]
    model_module_name = ".".join(model_class_parts[:-1])
    print("Loading {} function from {}".format(model_class_name, model_module_name))        
    try:
        image_function = getattr(importlib.import_module(model_module_name), model_class_name)
    except (ImportError, AttributeError) as e:
        # fallback: try loading from "rendering" subdirectory of library path (todo: default/enforce?)
        try:
            image_function = getattr(importlib.import_module("rendering." + model_module_name), model_class_name)
        except ImportError as e:
            helpful_interface_message_exit(fullname, e)
    print("function loaded.")
    return image_function    

def load_scoring_object(scoring_string):
    scoring_parts = scoring_string.split(":")
    fullname = scoring_parts[0]
    config_suffix = ""
    if len(scoring_parts) > 1:
        config_suffix = scoring_parts[1]
    model_class_name = "Scoring"
    model_module_name = fullname
    # print("Loading {} class from {}".format(model_class_name, model_module_name))
    try:
        scoring_class = getattr(importlib.import_module(model_module_name), model_class_name)
    except ImportError as e:
        try:
            # fallback: try loading from "scoring" subdirectory of library path (todo: default/enforce?)
            scoring_class = getattr(importlib.import_module("scoring." + model_module_name), model_class_name)
        except ImportError as e:
            helpful_interface_message_exit(fullname, e)
    # print("class loaded.")
    scoring_object = scoring_class(config_suffix)
    return scoring_object
