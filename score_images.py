# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import numpy as np
import warnings
import sys
import os
import glob
import json
import random

from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from PIL import Image
from braceexpand import braceexpand
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from peutils import get_active_models_from_arg, unpack_requested_networks
from classloader import ScoringInterface

def real_glob(rglob):
    glob_list = braceexpand(rglob)
    files = []
    for g in glob_list:
        files = files + glob.glob(g)
    return sorted(files)

def save_json_vectors(vectors, files, filename):
    """Story np array of vectors as json"""
    with open(filename, 'w') as outfile:
        json.dump(vectors.tolist(), outfile)

    file = open("{}.txt".format(filename),"w")
    for f in files:
        file.write("{}\n".format(f))
    file.close()

def read_json_vectors(filename):
    """Return np array of vectors from json sources"""
    vectors = []
    with open(filename) as json_file:
        json_data = json.load(json_file)
    for v in json_data:
        vectors.append(v)
    np_array = np.array(vectors)

    fname = "{}.txt".format(filename)
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    files = [x.strip() for x in content]
    print("Found {} vectors and {} files".format(np_array.shape, len(files)))
    return np_array, files

def update_closest_list(closest_list, dist, index, num_to_keep):
    cur_entry = [index, dist]
    added_entry = False
    if(len(closest_list) < num_to_keep):
        closest_list.append(cur_entry)
    elif (dist < closest_list[-1][1]):
        closest_list[-1] = cur_entry
    else:
        return closest_list

    # we made a change, re-sort
    closest_list.sort(key=lambda x: x[1])
    return closest_list

def clip_length(s, l):
    if(len(s) > l):
        s = "{}...".format(s[:l-3])
    return s

def plot_topk(filename, decoded, correct_classes, title, bgcolor):
    my_dpi = 200.0
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111)

    ax.set_title(title)
    
    fig.patch.set_facecolor(bgcolor)
    ax.set_facecolor(bgcolor)

    topprobs = [n[2] for n in decoded]
    labels = [clip_length(n[1],16).replace("'","") for n in decoded]
    labels_lowercase = [l.lower() for l in labels]
    clipped_correct_classes = map(lambda s: clip_length(s,16), correct_classes)
    # correct_class = clip_length(correct_class,16)
    num_bars = len(decoded)
    barlist = ax.bar(range(num_bars), topprobs)
    # if target_class in topk:
    #     barlist[topk.index(target_class)].set_color('r')
    for correct_class in clipped_correct_classes:
        if correct_class.lower() in labels_lowercase:
            # current_index = labels.index(correct_class)
            # print("GRAPH: {} in {}/{} with score {} at {}".format(correct_class, labels, topprobs, topprobs[current_index], current_index))
            barlist[labels_lowercase.index(correct_class)].set_color('g')
    # plt.sca(ax2)
    ax.set_ylim([0, 1.1])
    ax.set_xticks(range(num_bars))
    ax.set_xticklabels(labels, rotation=90)
    # ax.set_xticklabels(labels, rotation=60, ha='right')
    fig.set_size_inches(360/my_dpi, 300/my_dpi)
    # fig.subplots_adjust(bottom=0.2)
    fig.savefig(filename, bbox_inches='tight', dpi=my_dpi)

def get_topk(decoded, correct_classes):
    labels = [clip_length(n[1],16).replace("'","") for n in decoded]
    scores = [n[2] for n in decoded]
    clipped_correct_classes = map(lambda s: clip_length(s,16), correct_classes)
    target_score = None
    target_ord = None
    for correct_class in clipped_correct_classes:
        if correct_class in labels:
            current_index = labels.index(correct_class)
            if target_score is None or target_score < scores[current_index]:
                # print("{} in {}/{} with score {} at {}".format(correct_class, labels, scores, scores[current_index], current_index))
                target_score = scores[current_index]
                target_ord = current_index
    return target_ord, target_score

def string_replacement(decoded, label_replace):
    old_label, new_label = label_replace.split(":")
    for ix in range(len(decoded)):
        code = decoded[ix][0]
        label = decoded[ix][1]
        score = decoded[ix][2]
        # print("checking ", label, score)
        if(label == old_label):
            # print("found")
            decoded[ix] = (code, new_label, score)
    return decoded

# hack of get_topk which swaps ties
def tie_promotion(decoded, correct_classes):
    labels = [clip_length(n[1],16).replace("'","") for n in decoded]
    scores = [n[2] for n in decoded]
    clipped_correct_classes = map(lambda s: clip_length(s,16), correct_classes)
    target_score = None
    target_ord = None
    for correct_class in clipped_correct_classes:
        if correct_class in labels:
            current_index = labels.index(correct_class)
            if target_score is None or target_score < scores[current_index]:
                # print("{} in {}/{} with score {} at {}".format(correct_class, labels, scores, scores[current_index], current_index))
                target_score = scores[current_index]
                target_ord = current_index
    # now find the best ord with the same score...
    best_ord = target_ord
    for ix in range(len(scores)):
        if scores[ix] == target_score and ix < best_ord:
            # print("Found better tie_breaker {} < {}".format(ix, best_ord))
            best_ord = ix
        # for correct_class in clipped_correct_classes:
        #     if scores[ix] < target_score and correct_class in labels:
        #         print("WARNING: LOGIC BUG. SCORES < TARGET -> {}, {}", scores[ix], target_score)
        #     elif scores[ix] == target_score and ix < best_ord:
        #         print("Found better tie_breaker {} < {}", ix, best_ord)
        #         best_ord = ix
    # swap if necessary
    if best_ord != target_ord:
        # print("Swapping {} with {}", target_ord, best_ord)
        best_decoded = decoded[best_ord]
        decoded[best_ord] = decoded[target_ord]
        decoded[target_ord] = best_decoded
    return decoded

        # cur_decoded = []
        # for j in range(len(table)):
        #     cur_decoded.append((table[j][0], table[j][1], scores[i][j]))
        # all_decoded.extend(cur_decoded)

def decoded_from_table(table, scores, top=5):
    results = []
    # print(scores.shape)
    best =  np.argmax(scores[0])
    # print("DFT ", best)
    # print("DFT ", scores[0].shape)
    # print(f"DFT best score is {best} -> {table[best]} = {scores[0][best]}")
    # print(len(table), len(table[0]))
    for i in range(len(scores)):
        cur_score = scores[i]
        top_indices = cur_score.argsort()[-(top):][::-1]
        # top_indices = cur_score.argsort()[-(top+1):][:-1]
        # print(top_indices)
        result = [ (table[j][0], table[j][1], cur_score[j]) for j in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.extend(result)
    # print(results[:3])
    return results

graph_color_test='#FFFFFF'
graph_color_train1='#FFEB8D'
graph_color_train2='#FFF8DC'
graph_color_train3='#FFFCEE'

def main():
    parser = argparse.ArgumentParser(description="score images against classifiers")
    parser.add_argument('--input-glob', default='elephant.jpg',
                        help="inputs")
    parser.add_argument("--networks", default="standard",
                        help="comma separated list of networks")
    parser.add_argument("--train1", default=None,
                        help="comma separated list of train1 networks")
    parser.add_argument("--train2", default=None,
                        help="comma separated list of train2 networks")
    parser.add_argument("--train3", default=None,
                        help="comma separated list of train3 networks")
    parser.add_argument("--grade", default=None,
                        help="overlay grade")
    parser.add_argument("--label-replace", default=None,
                        help="replace a label")
    parser.add_argument('--num-closest', type=int, default=1,
                        help="number of closest to find/report")
    parser.add_argument('--topn', type=int, default=5,
                        help="number of entries to show in topN bar graphs")
    parser.add_argument('--show-prediction', default=False, action='store_true',
                        help="Print out prediction")
    parser.add_argument('--do-graphfile', default=False, action='store_true',
                        help="Make a new image file with graphs")
    parser.add_argument("--target-class", default=None,
                        help="target class for coloring output graph")
    parser.add_argument("--trim-prefix", default=None,
                        help="trim a prefix for a scoring module")
    parser.add_argument('--outfile', default=None,
                         help='path to where to put output file')
    parser.add_argument('--solo-outfile', default=False, action='store_true',
                        help="save csv version of (first) graph output")
    parser.add_argument('--graphfile-prefix', default="graph_",
                         help='prefix for graphfile')
    args = parser.parse_args()

    files = real_glob(args.input_glob)
    print("Found {} files in glob {}".format(len(files), args.input_glob))
    if len(files) == 0:
        print("No files to process")
        sys.exit(0)

    if args.solo_outfile:
        dirname = os.path.dirname(files[0])
        fname = os.path.basename(files[0])
        filebase = fname.rsplit('.',1)[0]
        # now paste it all together
        args.outfile = os.path.join(dirname, "{}{}.csv".format(args.graphfile_prefix,filebase))
        print("solo-outfile set to {}".format(args.outfile))

    active_models = get_active_models_from_arg(args.networks)
    active_model_keys = sorted(active_models.keys())

    trim_prefix_left = None
    if args.trim_prefix is not None:
        if ':' in args.trim_prefix:
            trim_prefix_left, trim_prefix_right = args.trim_prefix.split(':', 2)
        else:
            trim_prefix_left = "remove"
            trim_prefix_right = None

    train1_networks = []
    train2_networks = []
    train3_networks = []
    if args.train1 is not None:
        train1_networks = unpack_requested_networks(args.train1);
    if args.train2 is not None:
        train2_networks = unpack_requested_networks(args.train2);
    if args.train3 is not None:
        train3_networks = unpack_requested_networks(args.train3);

    if args.outfile is not None:
        outfile = open(args.outfile, 'w+')
        # write header
        outfile.write("path,")
        for k in active_model_keys:
            outfile.write("{}_class,{}_score,{}_target_ord,{}_target_score,".format(k, k, k, k))
        majority_class = outfile.write("consensus,low_score,product_score,majority_code,majority_class,majority_count")
        outfile.write("\n")
    else:
        outfile = None

    if len(files) > 1:
        files_iterator = tqdm(files)
    else:
        files_iterator = files

    reported_outfiles = []

    for img_path in files_iterator:
        if outfile is not None:
            outfile.write("{},".format(img_path))
        consensus_class = None
        low_score = 1.0
        product_score = 1.0
        top_codes = {}
        code_table = {}
        target_classes = []
        old_target_class = None
        if args.target_class is not None:
            target_class_strings = list(args.target_class.split(","))
            for target_class in target_class_strings:
                if target_class is not None and target_class.isdigit():
                  # TODO: refactor (this is from build_image.py)
                  class_file = os.path.expanduser("~/.keras/models/imagenet_class_index.json")
                  with open(class_file) as json_data:
                    d = json.load(json_data)
                  old_target_class = target_class
                  target_class = d[target_class][1].replace("'","")
                  # print("Resolved {} to {}".format(old_target_class, target_class))
                target_classes.append(target_class)

        pred_table = {}
        decoded_table = {}
        for k in active_model_keys:
            model = active_models[k]
            target_size = model.get_target_size()
            image_preprocessor = model.get_input_preprocessor()
            # print("So size and prep: ", target_size, image_preprocessor)
    
            img = image.load_img(img_path, target_size=target_size)
            # img.save(f"debug_this_{target_size}.png")
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            # print("X shape: ", x.shape)
            if image_preprocessor is not None:
                x = image_preprocessor(x)

            pred_table[k] = model.predict(x)

        for k in active_model_keys:
            preds = pred_table[k]

            # print(preds);
            nsfw_group = ['opennsfw', 'googlesafe', 'rekogmod', 'clarifai_nsfw']
            # print("looking for ", k, " in ", nsfw_group)
            if isinstance(preds,dict) and "decoded" in preds:
                # print(k,preds)
                decoded = preds['decoded']
            elif isinstance(preds,dict) and "table" in preds:
                # print(k,preds)
                decoded = decoded_from_table(preds["table"], preds["scores"], top=args.topn)
            elif k in nsfw_group:
                # print('preds: {}'.format(preds.shape))
                decoded = [
                    ('nsfw', 'nsfw', preds[0][1]),
                    ('sfw', 'sfw', preds[0][0]),
                ]
                # k = "open_nsfw"
            elif k.startswith("goog_"):
                # print('preds: {}'.format(preds.shape))
                decoded = [
                    (k, k, preds[0][0])
                ]
                # k = "open_nsfw"
            elif "face" in k:
                decoded = keras_vggface.utils.decode_predictions2(preds)[0]
            else:
                # print("NOTE: ", preds.shape, preds[0][0], preds[0][51])
                decoded = decode_predictions(preds, top=args.topn)[0]

            decoded_table[k] = decoded

        if len(target_classes) == 1 and target_classes[0] == "first":
            target_classes = [ decoded_table[active_model_keys[0]][0][1].replace("'","") ]

        elif len(target_classes) == 2 and target_classes[0] == "use":
            # print(target_classes[1])
            # print(decoded_table)
            # print(decoded_table[target_classes[1]])
            # print(decoded_table[target_classes[1]][0])
            target_classes = [ decoded_table[target_classes[1]][0][1].replace("'","") ]

        elif len(target_classes) == 1 and target_classes[0] == "vote":
            top_ones = []
            for d in decoded_table.values():
                if(len(d) > 0):
                    top_ones.append(d[0][1].replace("'",""))
            st = stats.mode(top_ones)
            target_classes = [ st[0][0] ]
            # print(top_ones, st, target_classes)

        if args.do_graphfile:
            bars_dir = "outputs/bars/{:05d}".format(random.randint(0,10000))
            if not os.path.exists(bars_dir):
                os.makedirs(bars_dir)

        for k in active_model_keys:
            decoded = decoded_table[k]
            model = active_models[k]

            if args.show_prediction:
                print('{} predicted: {}'.format(k, decoded))

            if hasattr(model, 'pos_labels'):
                cur_target_classes = target_classes + model.pos_labels
            else:
                cur_target_classes = target_classes.copy()

            # print(k, cur_target_classes, decoded)

            if trim_prefix_left is not None and trim_prefix_left == 'remove' and k.find(":") >= 0:
                model_suffix = k.split(":")[1]
            elif trim_prefix_left is not None and k.find(trim_prefix_left) >= 0:
                model_suffix = k.replace(trim_prefix_left, trim_prefix_right)
            else:
                # model_suffix = k.split(":")[0]
                model_suffix = k

            decoded = tie_promotion(decoded, cur_target_classes)
            if args.label_replace is not None:
                # print("replacing ", args.label_replace)
                decoded = string_replacement(decoded, args.label_replace)

            clean_k = k.replace("/","_")
            if args.do_graphfile:
                if k in train1_networks:
                    graph_color = graph_color_train1
                    prefix="01"
                elif k in train2_networks:
                    graph_color = graph_color_train2
                    prefix="02"
                elif k in train3_networks:
                    graph_color = graph_color_train3
                    prefix="03"
                else:
                    graph_color = graph_color_test
                    prefix="04"
                plot_topk("{}/bars_{}_{}.png".format(bars_dir, prefix, clean_k), decoded[:12], cur_target_classes, clip_length(model_suffix, 16), bgcolor=graph_color)

            if outfile is not None:
                if len(decoded) == 0:
                    print("Warning: no predictions. Using 'unknown' as placeholder")
                    top = "unknown", "unknown", "0.01"
                else:
                    top = decoded[0]
                p_code, p_class, p_score = top[0], top[1], top[2]
                target_ord, target_score = get_topk(decoded, cur_target_classes)
                outfile.write("{},{},{},{},".format(p_class, p_score,target_ord,target_score))
                code_table[p_code] = p_class
                if p_code in top_codes:
                    top_codes[p_code] = top_codes[p_code] + 1
                else:
                    top_codes[p_code] = 1
                if consensus_class is None:
                    consensus_class = p_class
                    low_score = p_score
                    product_score = p_score
                elif consensus_class != p_class:
                    consensus_class = "NONE"
                    low_score = 0.0
                    product_score = 0.0
                else:
                    product_score = product_score * p_score
                    if p_score < low_score:
                        low_score = p_score
        if outfile is not None:
            majority_code = max(top_codes, key=top_codes.get)
            majority_class = code_table[majority_code]
            majority_count = top_codes[majority_code]
            outfile.write("{},{},{},{},{},{}".format(consensus_class, low_score, product_score, majority_code, majority_class, majority_count))
            outfile.write("\n")
        if args.do_graphfile:
            raw_im = Image.open(img_path)
            width, height = raw_im.size
            dirname = os.path.dirname(img_path)
            fname = os.path.basename(img_path)
            # first make ?x3 graphs
            command = "montage -tile x3 -geometry +0+0 -gravity northeast {}/bars_*.png {}/triple.png".format(bars_dir,bars_dir)
            # this geometry resize all tiles to match: '1x1+0+0<'
            os.system(command)
            # resize to height
            command = "convert {}/triple.png -geometry x{} {}/triple.png".format(bars_dir,height,bars_dir)
            os.system(command)
            if args.grade is not None:
                grade_size = int(0.65 * height)
                command = "convert {}/triple.png -gravity Center -fill 'rgba(0,129,0,0.70)' -pointsize {} -font Helvetica-Bold -annotate 0 '{}' {}/triple.png".format(bars_dir, grade_size, args.grade, bars_dir)
                os.system(command)
            # make a blank space
            command = "convert -size 8x8 xc:black {}/blackimage.png".format(bars_dir)
            os.system(command)
            # now paste it all together
            graphfile = os.path.join(dirname, "{}{}".format(args.graphfile_prefix,fname))
            command = "montage -tile x1 -background black -geometry +0+0 {} {}/blackimage.png -gravity northeast {}/triple.png {}".format(img_path, bars_dir, bars_dir, graphfile)
            os.system(command)
            reported_outfiles.append(graphfile)
            for hgx in glob.glob("{}/*.png".format(bars_dir)):
                # print("REMOVING: {}".format(hgx))
                os.remove(hgx)
            os.rmdir(bars_dir)
    if outfile is not None:
        outfile.close()
    for report_file in reported_outfiles:
        print("{}".format("-> {}".format(report_file)))


if __name__ == '__main__':
    main()
