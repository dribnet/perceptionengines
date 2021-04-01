from classloader import load_scoring_object
import os
import json

def get_model_from_name(k):
  model = load_scoring_object(k)
  return model

model_groups = {
  "standard6,": "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception,",
  "standard9,": "standard6,inceptionresnetv2,nasnet,nasnetmobile,",
  "standard13,": "standard9,densenet121,densenet169,densenet201,mobilenetv2,",
  "standard18,": "standard13,resnet101,resnet152,resnet50v2,resnet101v2,resnet152v2,",
  "train1,": "vgg19,resnet50,inceptionv3,xception,",
  "standard,":  "standard6,",
  "all_,": "standard18,",
}

def unpack_models_string(models_string):
  # a messy way to do substiution of aliases. whatever.
  cur_models_string = ""
  next_models_string = models_string
  while cur_models_string != next_models_string:
    cur_models_string = next_models_string
    if not next_models_string.endswith(","):
      next_models_string = next_models_string + ","
    for key in model_groups:
      next_models_string = next_models_string.replace(key, model_groups[key])
    # print("how about ", cur_models_string, "becoming", next_models_string)
  return cur_models_string

def unpack_requested_networks(networks):
  networks = unpack_models_string(networks)
  requested_networks = networks.split(",")
  # remove empty strings
  requested_networks = [x for x in requested_networks if x]
  # remove duplicates and sort
  requested_networks = sorted(list(dict.fromkeys(requested_networks)))
  return requested_networks  

def get_active_models_from_arg(networks):
  requested_networks = unpack_requested_networks(networks)
  print("Requested networks: ", requested_networks)
  active_models = {}
  for k in requested_networks:
      if(not k.startswith("standard")):
          print("Setting up {}".format(k))
          active_models[k] = get_model_from_name(k)    
  if len(active_models) == 0:
      print("_____ WARNING: no active models ______")
  return active_models


# utilities for mapping imagenet names <-> indexes
def sanatize_label(l):
  l = l.lower()
  l = l.replace("'","")
  l = l.replace(" ", "_")
  return l

def open_class_mapping(filename="~/.keras/models/imagenet_class_index.json"):
  class_file = os.path.expanduser(filename)
  with open(class_file) as json_data:
    mapping = json.load(json_data)
  clean_mapping = {}
  for k in mapping:
    v = mapping[k]
    clean_key = int(k)
    clean_mapping[clean_key] = [sanatize_label(v[0]), sanatize_label(v[1])]
  return clean_mapping

def get_map_record_from_key(mapping, key):
  if isinstance(key, int):
    map_index = key
  elif key.isdigit():
    map_index = int(key)
  else:
    map_index = None
    clean_label = sanatize_label(key)
    # first try mapping the label to an index
    for k in mapping:
      if mapping[k][1] == clean_label and map_index is None:
        map_index = k
    if map_index is None:
      # backup try mapping the label to a fullname
      for k in mapping:
        if mapping[k][2] == clean_label and map_index is None:
          map_index = k
    if map_index is None:
      print("class mapping for {} not found", key)
      return None

  return [map_index, mapping[map_index][0], mapping[map_index][1]]

def get_class_index(mapping, key):
  map_record = get_map_record_from_key(mapping, key)
  if map_record is None:
    return None
  return map_record[0]

def get_class_fullname(mapping, key):
  map_record = get_map_record_from_key(mapping, key)
  if map_record is None:
    return None
  return map_record[1]

def get_class_label(mapping, key):
  map_record = get_map_record_from_key(mapping, key)
  if map_record is None:
    return None
  return map_record[2]

def get_class_index_list(mapping, keys):
  key_list = keys.split(",")
  index_list = [get_class_index(mapping, k) for k in key_list]
  return index_list
