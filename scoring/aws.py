# AWS Rekognition labels
# conda install -c conda-forge boto3

import os
import re
import io
from PIL import Image
import numpy as np
from classloader import ScoringInterface

import boto3
boto_client = None
min_api_score = 0.1

import sys

class Scoring(ScoringInterface): 
    pos_labels = []
    neg_labels = []
    def __init__(self, label):
        super(Scoring, self).__init__()

        global boto_client
        if boto_client is None:
          boto_client = boto3.client('rekognition','us-east-1')
        label_parts = re.split('([+-])', label)
        label_parts.pop(0)
        print("label_parts: {}".format(label_parts))
        for i in range(int(len(label_parts)/2)):
            which_list = label_parts[i*2]
            if which_list == '+':
                self.pos_labels.append(label_parts[i*2+1].lower())
            elif which_list == '-':
                self.neg_labels.append(label_parts[i*2+1].lower())
            else:
                print("Problem with this aws label, aborting: {} ({})".format(labels, which_list))
                sys.exit(1)
        self.square_score = None
        if 'AWS_SQUARE' in os.environ:
            try:
                self.square_score = int(os.environ['AWS_SQUARE'])
            except ValueError:
                self.square_score = 2
            print("Enabling AWS SQUARE_SCORE: {}".format(self.square_score))
        print("Setup AwsLabel {}".format(label))

    def predict(self, batch, explain=False):
        global boto_client
        all_values = []
        all_decoded = []

        # build lists of images at all needed sizes
        for im_pixels in batch:
            # print(im_pixels)
            img = Image.fromarray(im_pixels.astype('uint8'))
            imgByteArr = io.BytesIO()
            # todo: is png the best here?
            img.save(imgByteArr, format='jpeg', quality=95)
            im_bytes = imgByteArr.getvalue()
            response = boto_client.detect_labels(Image={'Bytes': im_bytes}, MinConfidence=0.0)
            labels = response['Labels']

            # print(labels)
            score_table = {}
            for l in labels:
              label_name = l['Name'].replace(" ","_").lower()
              score_table[label_name] = l['Confidence']/100.0
            # print("SCORE TABLE: {}", score_table)
            pos_score = 0.0
            if len(self.pos_labels) > 0:
              pos_score = 1.0;
              for l in self.pos_labels:
                if l in score_table:
                  chunk_score = score_table[l]
                else:
                  chunk_score = min_api_score
                pos_score = pos_score * chunk_score
            neg_score = 0.0
            if len(self.neg_labels) > 0:
              neg_score = 1.0;
              for l in self.neg_labels:
                if l in score_table:
                  chunk_score = score_table[l]
                else:
                  chunk_score = min_api_score
                neg_score = neg_score * chunk_score
                cur_score = 0.5 + 0.5 * pos_score - 0.5 * neg_score
            else:
              cur_score = pos_score

            if self.square_score is not None:
              for i in range(1,self.square_score):
                cur_score = cur_score * cur_score

            # print("score = {}, {}, {}".format(cur_score, pos_score, neg_score))
            decoded = []
            for l in labels:
              cur_label = l['Name'].replace(" ","_").lower()
              decoded.append(("aws_{}".format(cur_label), cur_label, l['Confidence'] / 100.0))
            all_values = all_values + [cur_score]
            all_decoded = all_decoded + decoded

        num_preds = len(all_values)
        preds = np.array([all_values]).reshape(num_preds, 1)
        # print("AWS SAYS", preds)
        return {"scores": preds, "decoded": all_decoded}

    def get_target_size(self):
        return None

    def get_input_preprocessor(self):
        return None

