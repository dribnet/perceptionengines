from classloader import load_scoring_object

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
  "all,": "standard18,",
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
