import tensorflow as tf
import numpy as np
from tmp.models import LSTM

class Predictor:
    def __init__(self):
        # Load the pre-trained model
        lstm = LSTM(n_classes=2)
        model = lstm.model
        x = np.random.random((1, 120, 7))
        tmp = model(x)
        model.load_weights("model/model.h5")
        self.model = model

    def predict(self, data):
        input_data = np.array(data)
        input_data = np.expand_dims(input_data, axis=0)
        # Make a prediction
        predictions = self.model.predict(input_data)
        return predictions


