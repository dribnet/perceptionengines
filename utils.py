from classloader import load_scoring_object

def get_model_from_name(k):
  model = load_scoring_object(k)
  return model

model_groups = {
  "standard":  "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception",
  "standard6": "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception"
}

def get_active_models_from_arg(networks):
  for key in model_groups:
    networks = networks.replace(key, model_groups[key])
  active_models = {}
  requested_networks = networks.split(",")
  # remove duplicates and sort
  requested_networks = sorted(list(dict.fromkeys(requested_networks)))
  # print("Requested networks: ", requested_networks)
  for k in requested_networks:
      if(not k.startswith("standard")):
          print("Setting up {}".format(k))
          active_models[k] = get_model_from_name(k)    
  if len(active_models) == 0:
      print("_____ WARNING: no active models ______")
  return active_models
