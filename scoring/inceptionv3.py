from classloader import ScoringInterface
import os
import keras

class Scoring(ScoringInterface): 
    def __init__(self, config):
        super(Scoring, self).__init__()
        self.model = keras.applications.inception_v3.InceptionV3()

    def predict(self, batch, explain=False):
        return self.model.predict(batch)

    def get_target_size(self):
        return (299, 299)

    def get_input_preprocessor(self):
        return keras.applications.inception_v3.preprocess_input

