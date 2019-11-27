from classloader import load_scoring_object

def get_model_from_name(k):
  model = load_scoring_object(k)
  return model

model_groups = {
  "standard,":  "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception,",
  "standard6,": "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception,",
  "standard9,": "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception,inceptionresnetv2,nasnet,nasnetmobile,",
  "standard13,": "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception,inceptionresnetv2,nasnet,nasnetmobile,densenet121,densenet169,densenet201,mobilenetv2,",
  "standard18,": "vgg16,vgg19,mobilenet,resnet50,inceptionv3,xception,inceptionresnetv2,nasnet,nasnetmobile,densenet121,densenet169,densenet201,mobilenetv2,resnet101,resnet152,resnet50v2,resnet101v2,resnet152v2,",
}

def get_active_models_from_arg(networks):
  # a messy way to do substiution of aliases. whatever.
  if not networks.endswith(","):
    networks = networks + ","
  for key in model_groups:
    networks = networks.replace(key, model_groups[key])
  active_models = {}
  requested_networks = networks.split(",")
  # remove empty strings
  requested_networks = [x for x in requested_networks if x]
  # remove duplicates and sort
  requested_networks = sorted(list(dict.fromkeys(requested_networks)))
  print("Requested networks: ", requested_networks)
  for k in requested_networks:
      if(not k.startswith("standard")):
          print("Setting up {}".format(k))
          active_models[k] = get_model_from_name(k)    
  if len(active_models) == 0:
      print("_____ WARNING: no active models ______")
  return active_models
