import argparse
import sys
import math
import importlib
from PIL import Image, ImageDraw
import numpy as np

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
    except ImportError as e:
        # fallback: try loading from "renderer" subdirectory of library path (todo: default/enforce?)
        try:
            image_function = getattr(importlib.import_module("renderer." + model_module_name), model_class_name)
        except ImportError as e:
            helpful_interface_message_exit(fullname, e)
    print("function loaded.")
    return image_function    
