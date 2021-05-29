import argparse
import glob
import sys
import os
import datetime
import numpy as np
from classloader import load_image_function
from braceexpand import braceexpand
import random
import io
from PIL import Image

def real_glob(rglob):
    glob_list = braceexpand(rglob)
    files = []
    for g in glob_list:
        files = files + glob.glob(g)
    return sorted(files)

# this function can fill in placeholders for %DATE%, %SIZE% and %SEQ%
def emit_filename(filename, template_dict):
    datestr = datetime.datetime.now().strftime("%Y%m%d")
    filename = filename.replace('%DATE%', datestr)

    for key in template_dict:
        pattern = "%{}%".format(key)
        value = "{}".format(template_dict[key])
        filename = filename.replace(pattern, value)

    if '%SEQ%' in filename:
        # determine what the next available number is
        cur_seq = 1
        candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        while os.path.exists(candidate):
            cur_seq = cur_seq + 1
            candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        filename = candidate
    return filename

def save_file_or_files(infile, im, outfile, template_dict):
    # allow list of outputs (layers)
    if isinstance(im, list):
        for ix in range(len(im)):
            outfile_name = outfile.format(ix+1)
            emitted_filename = emit_filename(outfile_name, template_dict)
            im[ix].save(emitted_filename)
            print("{:7s} -> {}".format(infile, emitted_filename))
    else:
        emitted_filename = emit_filename(outfile, template_dict)
        im.save(emitted_filename)
        print("{:7s} -> {}".format(infile, emitted_filename))

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="render dopes image")
    parser.add_argument('--input-glob', default=None,
                        help="inputs")
    parser.add_argument('--outfile', default="outputs/%RENDERER%_%DATE%_l%LEN%_r%SEED%_s%SIZE%_%SEQ%.png",
                        help="single output file")
    parser.add_argument('--outbase', default=None,
                         help='basename for the output file')
    parser.add_argument('--renderer', default='lines1',
                        help="(none)")
    parser.add_argument('--size', default=1200, type=int,
                        help='(none)')
    parser.add_argument('--random-seed', default=None, type=int,
                        help='Use a specific random seed (for repeatability)')
    parser.add_argument('--interpolate', default=None, type=int,
                        help='Turn on interpolation and give a number of frames of output')
    parser.add_argument('--versions', default=1, type=int,
                        help='how many versions to make [put formatter in filename if > 1]')
    parser.add_argument('--length', default=None, type=int,
                        help='Length of generated vector list')
    args = parser.parse_args()

    template_dict = {}

    array_to_image = load_image_function(args.renderer + ".render")    
    render_parts = args.renderer.split('.')
    template_dict["RENDERER"] = render_parts[-1]

    if args.random_seed is not None:
        print("Setting random seed: ", args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        template_dict["SEED"] = args.random_seed
    else:
        template_dict["SEED"] = None

    template_dict["SIZE"] = args.size

    if args.outfile == "None":
        args.outfile = None
      # for i in range(args.random_seed):
      #   n = np.random.uniform()

    if args.input_glob is None:
        files = ["(random)"]
    else:
        files = real_glob(args.input_glob)
        print("Found {} files in glob {}".format(len(files), args.input_glob))
        if len(files) == 0:
            print("No files to process")
            sys.exit(0)

    if args.interpolate is not None:
        if len(files) != 2:        
            print("Interpolate is brittle and needs exactly two files")
            sys.exit(0)
        input_array1 = np.load(files[0])
        print("Loaded {}: shape {}".format(files[0], input_array1.shape))
        input_array2 = np.load(files[1])
        print("Loaded {}: shape {}".format(files[1], input_array2.shape))
        if len(input_array1) != len(input_array2):
            print("Interpolate is brittle and files are not equal length: {}, {}".format(len(input_array1), len(input_array2)))
            sys.exit(0)
        for i in range(args.interpolate):
            frac = i / (args.interpolate - 1)
            interp_array = lerp(frac, input_array1, input_array2)
            im = array_to_image(interp_array, args.size)
            outfile = args.outfile.format(i+1)
            save_file_or_files("interpolation {}".format(i+1), im, outfile, template_dict)
        sys.exit(0)

    cur_file_num = 1
    for infile in files:
        if infile == "(random)":
            if args.length is None:
                length = 24
            else:
                length = args.length
            template_dict["LEN"] = length
            input_array = np.random.uniform(low=0.02, high=0.98, size=(length, 8))
            print("Created random input with shape {}".format(input_array.shape))
        else:
            input_array = np.load(infile)
            print("Loaded {}: shape {}".format(infile, input_array.shape))
            if args.length is not None:
                input_array = input_array[:args.length]
            template_dict["LEN"] = len(input_array)

        for n in range(args.versions):
            im = array_to_image(input_array, args.size)
            if args.outbase is not None:
                dirname = os.path.dirname(infile)
                outfile = os.path.join(dirname, args.outbase)
            else:
                outfile = args.outfile

            # somewhat messy handling of list case twice for file naming
            if not isinstance(im, list):
                outfile = outfile.format(cur_file_num)
                cur_file_num = cur_file_num + 1
            save_file_or_files(infile, im, outfile, template_dict)
