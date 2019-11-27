# Google vision labels
# conda install -c conda-forge google-cloud-vision

import os
import re
import io
from PIL import Image
import numpy as np
from classloader import ScoringInterface

from google.cloud import vision as vision
google_client = None

import sys

class Scoring(ScoringInterface): 
    pos_labels = []
    neg_labels = []
    def __init__(self, label):
        super(Scoring, self).__init__()

        global google_client
        if google_client is None:
          google_client = vision.ImageAnnotatorClient()
        label_parts = re.split('([+-])', label)
        label_parts.pop(0)
        print("label_parts: {}".format(label_parts))
        for i in range(int(len(label_parts)/2)):
            which_list = label_parts[i*2]
            target_class = label_parts[i*2+1].replace(" ","_").lower()
            if which_list == '+':
                self.pos_labels.append(target_class)
            elif which_list == '-':
                self.neg_labels.append(target_class)
            else:
                print("Problem with this google label, aborting: {} ({}, {})".format(labels, which_list, target_class))
                sys.exit(1)
        self.square_score = None
        # can be used as a multiplier
        if 'GOOGLE_SQUARE' in os.environ:
            try:
                self.square_score = int(os.environ['GOOGLE_SQUARE'])
            except ValueError:
                self.square_score = 2
            print("Enabling GOOGLE SQUARE_SCORE: {}".format(self.square_score))
        print("Setup GoogLabel {}".format(label))

    def predict(self, batch, explain=False):
        global google_client
        all_values = []
        all_decoded = []

        batch_size = 16
        batch_index = 0
        while batch_index < len(batch):
            wa_batch = batch[batch_index:batch_index+batch_size]
            batch_index += batch_size

            requests = []
            batch_scores = []
            batch_decoded = []

            # build lists of images at all needed sizes
            for im_pixels in wa_batch:
                # print("PIXSHAPE: ", im_pixels.shape)
                img = Image.fromarray(im_pixels.astype('uint8'))
                imgByteArr = io.BytesIO()
                # todo: is png the best here?
                img.save(imgByteArr, format='jpeg', quality=95)
                # img.save("/tmp/test.jpg", format='jpeg', quality=95)
                imgByteArr = imgByteArr.getvalue()
                gimage = vision.types.Image(content=imgByteArr)
                request = dict(
                    image=gimage,
                    features=[{'type': vision.enums.Feature.Type.LABEL_DETECTION}]
                    )
                requests.append(request)

            batch_api_result = google_client.batch_annotate_images(requests)
            cur_scores = []
            cur_decoded = []
            batch_responses = [f for f in batch_api_result.responses]
            for response in batch_responses:
              labels = response.label_annotations
              score_table = {}
              for l in labels:
                label_name = l.description.replace(" ","_").lower()
                score_table[label_name] = l.score
              # print("SCORE TABLE: {}", score_table)
              pos_score = 0.0
              if len(self.pos_labels) > 0:
                pos_score = 1.0;
                for l in self.pos_labels:
                  if l in score_table:
                    chunk_score = score_table[l]
                    # print("+CHUNK {}, {}", l, chunk_score)
                  else:
                    chunk_score = min_api_score
                    # print("+CHUNK {}, {}", l, 0)
                  pos_score = pos_score * chunk_score
              neg_score = 0.0
              if len(self.neg_labels) > 0:
                neg_score = 1.0;
                for l in self.neg_labels:
                  if l in score_table:
                    chunk_score = score_table[l]
                    # print("-CHUNK {}, {}", l, chunk_score)
                  else:
                    chunk_score = min_api_score
                    # print("-CHUNK {}, {}", l, 0)
                  neg_score = neg_score * chunk_score
                cur_score = 0.5 + 0.5 * pos_score - 0.5 * neg_score
              else:
                cur_score = pos_score

              if self.square_score is not None:
                for i in range(1,self.square_score):
                  cur_score = cur_score * cur_score

              cur_scores = cur_scores + [cur_score]
              # print("score = {}, {}, {}".format(cur_score, pos_score, neg_score))
              decoded = []
              for l in labels:
                cur_label = l.description.replace(" ","_").lower()
                decoded.append(("goog_{}".format(cur_label), cur_label, l.score))
            batch_scores = batch_scores + cur_scores
            batch_decoded = batch_decoded + decoded
            # print("Scores: ", batch_scores)
            all_values = all_values + batch_scores
            all_decoded = all_decoded + batch_decoded

        num_preds = len(all_values)
        preds = np.array([all_values]).reshape(num_preds, 1)
        # print("GOOG SAYS", preds)
        return {"scores": preds, "decoded": all_decoded}

    def get_target_size(self):
        return None

    def get_input_preprocessor(self):
        return None

