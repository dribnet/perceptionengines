import argparse
import sys
import math
from PIL import Image, ImageDraw
import numpy as np
import keras
from keras.preprocessing import image
from collections import defaultdict

import json
import os
import random
import tensorflow as tf

from utils import get_active_models_from_arg
from classloader import ScoringInterface, load_image_function

model_inceptionv3 = None
model_vgg16 = None
model_resnet50 = None

render_size = 512

do_all = False
cur_cat_index = None
# default blacklist
cat_blacklist = [
  419,  # Band_Aid
  644,  # matchstick
  714,  # pick
  723,  # pinwheel
  767,  # rubber_eraser
  813,  # spatula
  920,  # traffic_light
  929   # ice_lolly
]

# https://stackoverflow.com/a/2290995/1010653
def leaders(xs):
  counts = defaultdict(int)
  for x in xs:
    counts[x] += 1
  print(counts)
  return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])

def read_category_blacklist(filename, max_thresh):
  global cat_blacklist

  cat_blacklist = []
  if not os.path.exists(filename):
    print("blacklist {} not found, continuing".format(filename))
    return

  with open(filename) as f:
    content = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  lines = [x.strip() for x in content]
  line_entries = []
  for line in lines:
    # remove comments
    line = line.partition('#')[0]
    line = line.rstrip()
    if len(line) > 0:
      line_entries.append(int(line))

  print("Found {} blacklist entries".format(len(line_entries)))
  sorted_pairs = leaders(line_entries)
  print(line_entries, sorted_pairs)
  cur_candidate = 0
  while len(sorted_pairs) > cur_candidate and sorted_pairs[cur_candidate][1] >= max_thresh:
    cat_blacklist.append(sorted_pairs[cur_candidate][0])
    cur_candidate = cur_candidate + 1
  print("final blacklist has {} entries ({})".format(len(cat_blacklist), cat_blacklist))

def append_category_blacklist(filename):
  with open(filename, "a") as f:
    imagenet_key = "{}".format(cur_cat_index)
    if imagenet_key in imagenet_classes:
      imagenet_name = imagenet_classes[imagenet_key][1]
      imagenet_name = imagenet_name.replace("'", "")
    else:
      imagenet_name = "category_{:04d}".format(int(imagenet_key))
    f.write("{} # {}\n".format(cur_cat_index, imagenet_name))

# closure around function to grab variables
def get_optimization_function(active_models, imagenet_indexes, array_to_image_fn, render_size):
  do_score_reverse = False
  if 'MODEL_REVERSE' in os.environ:
    print("-> predictions reversed")  
    do_score_reverse = True

  def f_optimize(wa):
    active_model_keys = sorted(active_models.keys())

    # build a table indexed by target_size for all resized image lists
    target_size_table = {}
    for k in active_model_keys:
      model = active_models[k]
      if isinstance(model, ScoringInterface):
        target_size = model.get_target_size()        
      else:
        target_size = get_target_size_from_name(k)
      target_size_table[target_size] = []

    # build lists of images at all needed sizes
    for w in wa:
      img = array_to_image_fn(w, render_size)
      for target_size in target_size_table:
        if target_size is None:
          imr = img
        else:
          imr = img.resize(target_size, resample=Image.BILINEAR)
        target_size_table[target_size].append(image.img_to_array(imr))

    # for k in target_size_table:
    #   print("_____+++", k, len(target_size_table[k]))
    #   for n in target_size_table[k]:
    #     print("_____>>>", k, n.shape)

    # # convert all lists to np arrays
    for target_size in target_size_table:
      target_size_table[target_size] = np.array(target_size_table[target_size])
      # print("SHAPE: ", target_size_table[target_size].shape)

    # make all predictions
    full_predictions = []
    for k in active_model_keys:
      model = active_models[k]
      if isinstance(model, ScoringInterface):
        target_size = model.get_target_size()        
        image_preprocessor = model.get_input_preprocessor()
      else:
        target_size = get_target_size_from_name(k)
        image_preprocessor = get_input_processor_from_name(k)

      # images = target_size_table[target_size]
      images = np.copy(target_size_table[target_size])
      batch = image_preprocessor(images)
      preds = model.predict(batch)
      # print("PREDS:", preds.shape, preds)
      if isinstance(preds,dict) and "scores" in preds:
        worthy = preds['scores']
      else:
        worthy = preds[:,imagenet_indexes]
      # print("Worthy: {}".format(np.array(worthy).shape))
      full_predictions.append(worthy)

    # convert predictions to np array
    full_predictions = np.array(full_predictions)
    if do_score_reverse:
      print("-> Applying predictions reversed")
      full_predictions = 1.0 - full_predictions
    top_classes = np.argmax(full_predictions,axis=2).flatten()
    top_class = np.argmax(np.bincount(top_classes))
    imagenet_index = imagenet_indexes[top_class]

    prediction_list = np.sum(full_predictions, axis=2)

    # extract rewards and merged
    rewards = np.prod(prediction_list, axis=0)
    merged = np.dstack(prediction_list)[0]
    return rewards, [imagenet_index, merged]

  return f_optimize

# closure around function to grab variables
def get_optimization_function_noindex(active_models, array_to_image_fn, render_size):
  def f_optimize(wa):
    active_model_keys = sorted(active_models.keys())

    # build a table indexed by target_size for all resized image lists
    target_size_table = {}
    for k in active_model_keys:
      model = active_models[k]
      if isinstance(model, ScoringInterface):
        target_size = model.get_target_size()        
      else:
        target_size = get_target_size_from_name(k)
      target_size_table[target_size] = []

    # build lists of images at all needed sizes
    for w in wa:
      img = array_to_image_fn(w, render_size)
      for target_size in target_size_table:
        imr = img.resize(target_size, resample=Image.BILINEAR)
        target_size_table[target_size].append(image.img_to_array(imr))

    # convert all lists to np arrays
    for target_size in target_size_table:
      target_size_table[target_size] = np.array(target_size_table[target_size])

    # which indeices are allowed
    all_elements = list(range(1000))
    # all_elements = list(range(8631))
    allowed = np.array([x for x in all_elements if x not in cat_blacklist])

    # make all predictions
    full_predictions = []
    for k in active_model_keys:
      model = active_models[k]
      if isinstance(model, ScoringInterface):
        target_size = model.get_target_size()        
      else:
        target_size = get_target_size_from_name(k)
      target_size = get_target_size_from_name(k)
      images = np.copy(target_size_table[target_size])
      # images = target_size_table[target_size]
      image_preprocessor = get_input_processor_from_name(k)
      batch = image_preprocessor(images)
      preds = model.predict(batch)
      worthy = preds[:,allowed]
      full_predictions.append(worthy)

    # convert predictions to np array
    full_predictions = np.array(full_predictions)
    top_classes = np.argmax(full_predictions,axis=2).flatten()
    top_class = np.argmax(np.bincount(top_classes))
    imagenet_index = allowed[top_class]

    prediction_list = full_predictions[:,:,top_class]

    # extract rewards and merged
    rewards = np.prod(prediction_list, axis=0)
    merged = np.dstack(prediction_list)[0]
    return rewards, [imagenet_index, merged]

  return f_optimize

# hyperparameters
sigma = 0.01 # noise standard deviation
alpha = 0.0005 # learning rate
good_enough = 0.999
max_dry_period = 30
imagenet_classes = None

do_freeze_hack = None

def optimize(outdir, array_to_image, f, iterations=1000, numpop=100, preview_size=512, num_lines=13, init_size=6, init_step=4, initial_array=None, rand_head=None, head_length=2):
  global imagenet_classes
  global cur_cat_index
  global do_freeze_hack

  if imagenet_classes == None:
    class_file = os.path.expanduser("~/.keras/models/imagenet_class_index.json")
    with open(class_file) as json_data:
      imagenet_classes = json.load(json_data)

  if do_freeze_hack is None:
      do_freeze_hack = False
      if 'FREEZE_HACK' in os.environ:
        print("-> freezing head and column 0")  
        do_freeze_hack = True

  # start the optimization
  best = None;

  # old - this looks F-ed up
  # w = np.random.normal(0.5, 0.3, size=(num_circles,8))
  # w = np.clip(w, 0.02, 0.98)
 
  if initial_array is None:
    # our initial guess is random
    # tried 0.3, 0.03, 0.1, 0.06
    num_circles = num_lines

    if init_size > num_circles:
      init_size = num_circles

    # step one: do a large batch and then find best
    w_try = np.random.uniform(low=0.02, high=0.98, size=(numpop, init_size, 8))
    rewards, _ = f(w_try)
    best_index = np.argmax(rewards)
    w_best = w_try[best_index]
    last_best_reward = rewards[best_index]
    # print(f"{rewards[best_index]} ({best_index}) is the best out of {rewards}")
    w = np.clip(w_best, 0.02, 0.98)
    init_count = 0
    im = array_to_image(w, size=preview_size)
    im.save("{}/init_{:05d}.jpg".format(outdir,init_count))
    init_count = init_count + 1

    while len(w) < num_circles:
      cur_w_size = len(w)
      next_w_size = cur_w_size + init_step
      if next_w_size > num_circles:
        next_w_size = num_circles

      w_try = np.random.uniform(low=0.02, high=0.98, size=(numpop, next_w_size, 8))
      for j in range(numpop):
        w_try[j, 0:cur_w_size] = w

      w_try = np.clip(w_try, 0.01, 0.99)

      # hack: enable random head
      if rand_head is not None and rand_head > 0:
        print("RUNNING RANDOM HEAD FOR INIT ({})".format(rand_head))
        num_shuf = int(numpop/2)
        w_try[:num_shuf,0:head_length,:] = np.random.uniform(low=0.02, high=0.98, size=(num_shuf, head_length, 8))

      rewards, _ = f(w_try)
      best_index = np.argmax(rewards)
      w_best = w_try[best_index]
      best_reward = rewards[best_index]
      best_change = (best_reward - last_best_reward) / last_best_reward
      best_change = "{:7.2f}%".format(100*best_change)
      last_best_reward = best_reward
      # print(f"init {i}: {rewards[best_index]} ({best_index}) is the best out of {rewards}")
      print("init {}: {:4.10f} ({})".format(init_count, best_reward, best_change))
      w = np.clip(w_best, 0.01, 0.99)
      im = array_to_image(w, size=preview_size)
      im.save("{}/init_{:05d}.jpg".format(outdir,init_count))
      init_count = init_count + 1
  else:
    w = initial_array
    num_circles = len(w)
  # rewards, diagnostics = f([w])
  # print("Sanity check: {} {}".format(rewards, diagnostics))

  # w = old_w
  im = array_to_image(w, size=preview_size)
  im.save("{}/start.png".format(outdir))
  np.save("{}/start".format(outdir), w)
  cycles_since_best = 0
  for i in range(iterations):
    im = array_to_image(w, size=preview_size)
    im.save("{}/epoch_{:05d}.jpg".format(outdir,i))
    # rewards, extra_information = f([w])
    rewards, extra_information = f([w, w, w, w, w, w, w, w, w, w])
    # rewards, extra_information = f([w, w, w, w, w])
    imagenet_class, diagnostics = extra_information
    imagenet_key = "{}".format(imagenet_class)
    if imagenet_key in imagenet_classes:
      imagenet_name = imagenet_classes[imagenet_key][1]
      imagenet_name = imagenet_name.replace("'", "")
    else:
      imagenet_name = "category_{:04d}".format(int(imagenet_key))
    # print(rewards.shape)
    # print(diagnostics.shape)
    r = np.mean(rewards, axis=0)
    r3 = list(100.0*np.mean(diagnostics,axis=0))
    # r, r3 = rewards[0], list(100.0*diagnostics[0])
    # print(rewards, diagnostics)
    # print(r, r3)
    is_best = " "
    if best is None or r > best:
      best = r
      cur_cat_index = imagenet_class
      im.save("{}/best.png".format(outdir))
      np.save("{}/best".format(outdir), w)
      file = open("{}/score.txt".format(outdir),"w")
      file.write("{:4.10f}\n".format(100.0*best))
      file.close()
      file = open("{}/category_index.txt".format(outdir),"w")
      file.write("{}\n".format(imagenet_class))
      file.close()
      file = open("{}/category.txt".format(outdir),"w")
      file.write("{}\n".format(imagenet_name))
      file.close()
      is_best = "*"
      cycles_since_best = 0
    else:
      cycles_since_best = cycles_since_best + 1
    print("iter {:05d} {}/{} reward: {:4.10f} {} {}".format(i, imagenet_class, imagenet_name, 100.0*r, r3, is_best))

    if best >= good_enough or cycles_since_best >= max_dry_period or i == iterations -1:
      im.save("{}/final.png".format(outdir))
      np.save("{}/final".format(outdir), w)
      if best >= good_enough:
        print("Early stop - quality threshold reached: {} > {}".format(best, good_enough))
      if cycles_since_best >= max_dry_period:
        print("Early stop - {} iterations without quality improvement".format(max_dry_period))
      else:
        print("The end - {} iterations reached".format(iterations))        
      # stop the loop
      break

    # initialize memory for a population of w's, and their rewards
    # samples from a normal distribution N(0,X)
    N = np.random.normal(0, 1.0, size=(numpop, num_circles, 8))

    if do_freeze_hack:
      print("Applying freeze_hack")
      N[0:head_length,:] = 0
      N[:,0] = 0
  
    # N = np.clip(N, -0.3, 0.3)
    # N[0] = np.zeros([num_circles, 8])
    R = np.zeros(numpop)
    w_try = np.random.normal(0, 1.0, size=(numpop, num_circles, 8))

    w_try = w + sigma * N
    w_try = np.clip(w_try, 0.01, 0.99)

    # hack: enable random head
    if rand_head is not None and i < rand_head:
      print("RUNNING RANDOM HEAD FOR ITERATION {}/{}".format(i,rand_head))
      num_shuf = int(numpop/2)
      w_try[:num_shuf,0:head_length,:] = np.random.uniform(low=0.02, high=0.98, size=(num_shuf, head_length, 8))

    R, _ = f(w_try)

    # standardize the rewards to have a gaussian distribution
    variation = np.std(R)
    if variation == 0:
      print("warning, no variation")
      A = (R - np.mean(R))
    else:
      A = (R - np.mean(R)) / np.std(R)

    # perform the parameter update. The matrix multiply below
    # is just an efficient way to sum up all the rows of the noise matrix N,
    # where each row N[j] is weighted by A[j]
    dot = np.dot(N.T, A)
    scaled_dot = alpha/(numpop*sigma) * dot

    if do_freeze_hack:
      print("Applying freeze_hack on w")
      w2 = w + scaled_dot.T
      w2[0:head_length,:] = w[0:head_length,:]
      w2[:,0] = w[:,0]
      w = w2
    else:
      w = w + scaled_dot.T

    if rand_head is not None and i < rand_head:
      print("COMMITTING RANDOM HEAD FOR ITERATION {}/{}".format(i,rand_head))
      best_index = np.argmax(R)
      w_best = w_try[best_index]
      w[0:2] = w_best[0:2]

    w = np.clip(w, 0.01, 0.99)

def main():
    global good_enough, max_dry_period, render_size
    global sigma, alpha

    parser = argparse.ArgumentParser(description="shape optimization")
    parser.add_argument('--input-array', default=None,
                        help="inputs")
    parser.add_argument('--outdir', default=None,
                        help="saved outputs")
    parser.add_argument('--catlog', default=None,
                        help="read/write to category log file")
    parser.add_argument('--maxcats', default=1,
                        help="maximum entries in catlog before category is blacklisted")
    parser.add_argument('--imagenet-index', default=None,
                        help='which imagenet index to optimize')
    parser.add_argument('--show-name', default=False, action='store_true',
                        help="show imagenet classname and exit")
    parser.add_argument('--show-friendly-name', default=False, action='store_true',
                        help="show imagenet classname and exit")
    parser.add_argument("--renderer", default="lines1",
                        help="renderer with image drawing function")
    parser.add_argument("--networks", default="all",
                        help="comma separated list of networks")
    parser.add_argument('--random-seed', default=None, type=int,
                        help='Use a specific random seed (for repeatability)')
    parser.add_argument('--random-head', default=None, type=int,
                        help='Add N steps of random initializations of head data (cur 2)')
    parser.add_argument('--header-length', default=2, type=int,
                        help='The length of the header (used for random-head)')
    parser.add_argument('--early-stop', default=None,
                        help='early stop number (good enough)')
    parser.add_argument('--max-attempts', default=30, type=int,
                        help='stop if no improvement for n cycles')
    parser.add_argument('--num-lines', default=17, type=int,
                        help='Number of lines to use')
    parser.add_argument('--render-size', default=None, type=int,
                        help='Size to render during testing')
    parser.add_argument('--num-pop', default=100, type=int,
                        help='Population size')
    parser.add_argument('--alpha-scale', default=1, type=float,
                        help='scale learning rate')
    parser.add_argument('--sigma-scale', default=1, type=float,
                        help='Scale random noise added each cycle')
    parser.add_argument('--init-step', default=4, type=int,
                        help='Init step')
    parser.add_argument('--max-iterations', default=1000, type=int,
                        help='Maximum iterations')
    args = parser.parse_args()

    # apply arguments
    outdir = args.outdir
    if args.early_stop is not None and args.early_stop.lower() != "none":
      good_enough = float(args.early_stop)
    max_dry_period = args.max_attempts
    print("Threshold is {} attempts to {}".format(max_dry_period, good_enough))

    if args.render_size is not None:
      render_size = args.render_size
      print("Overriding render_size to {}".format(render_size))

    if args.imagenet_index is not None and args.imagenet_index.isdigit():
      imagenet_indexes = [int(args.imagenet_index)]
    elif args.imagenet_index == "none":
      imagenet_indexes = None
    else:
      imagenet_indexes = list(map(int,args.imagenet_index.split(",")))

    # scale alpha and/or sigma
    if args.sigma_scale != 1:
      old_sigma = sigma
      sigma *= args.sigma_scale
      print("Scaling sigma {}x from {} to {}".format(args.sigma_scale, old_sigma, sigma))
    if args.alpha_scale != 1:
      old_alpha = alpha
      alpha *= args.alpha_scale
      print("Scaling alpha {}x from {} to {}".format(args.alpha_scale, old_alpha, alpha))
    # let's get to it
    if imagenet_indexes is not None:
      if args.networks == "vggface":
        with open("labels.json") as json_data:
          label_index = imagenet_indexes[0]
          d = json.load(json_data)
        # TODO: maybe handle multiples here
        categories = []
        if args.show_name or args.show_friendly_name:
          print(d[label_index].strip())
          sys.exit(0)
        if label_index < len(d):
          categories.append(d[label_index].strip())
        else:
          categories.append("face_{:04d}".format(int(label_index)))
      else:
        class_file = os.path.expanduser("~/.keras/models/imagenet_class_index.json")
        with open(class_file) as json_data:
          d = json.load(json_data)
        categories = []
        for imagenet_index in imagenet_indexes:
          imagenet_key = "{}".format(imagenet_index)
          friendly_name = d[imagenet_key][1]
          friendly_name = friendly_name.replace("'","")
          if args.show_name:
            print(d[imagenet_key][0])
            sys.exit(0)
          if args.show_friendly_name:
            print(friendly_name)
            sys.exit(0)
          if imagenet_key in d:
            categories.append(friendly_name)
          else:
            categories.append("category_{:04d}".format(int(imagenet_key)))
      print("----> Processing {}".format(categories))

    # make output directory if needed
    if outdir != '' and not os.path.exists(outdir):
      os.makedirs(outdir)

    # setup models
    active_models = get_active_models_from_arg(args.networks)

    array_to_image = load_image_function(args.renderer + ".render")    

    if args.random_seed:
      print("Setting random seed: ", args.random_seed)
      random.seed(args.random_seed)
      np.random.seed(args.random_seed)
      # TODO: not do this or maybe there is a tf2 way?
      tf.compat.v1.set_random_seed(args.random_seed)

    if args.catlog == "none":
      args.catlog = None

    if imagenet_indexes is None and args.catlog is not None:
      read_category_blacklist(args.catlog, args.maxcats)
    if imagenet_indexes is None:
      objective_fn = get_optimization_function_noindex(active_models, array_to_image, render_size)
    else:
      objective_fn = get_optimization_function(active_models, imagenet_indexes, array_to_image, render_size)
    # optimize(outdir, objective_fn, args.num_pop, 1000)
    if args.input_array is not None:
        initial_array = np.load(args.input_array)
        print("loaded data from: {}".format(args.input_array))
    else:
        initial_array = None
    # print("RANDOM HEAD {}".format(args.random_head))
    optimize(outdir, array_to_image, objective_fn, iterations=args.max_iterations, numpop=args.num_pop, preview_size=render_size, num_lines=args.num_lines, initial_array=initial_array, init_step=args.init_step, rand_head=args.random_head, head_length=args.header_length)
    if imagenet_indexes is None and args.catlog is not None:
      append_category_blacklist(args.catlog)


if __name__ == '__main__':
  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth=True
  # sess = tf.Session(config=config)
  main()
