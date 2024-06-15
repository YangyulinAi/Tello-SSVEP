import tensorflow as tf
import numpy as np
from tmp.models import LSTM

class Predictor:
    def __init__(self, model_path, buffer_size=120):
        # Load the pre-trained model
        lstm = LSTM(n_classes=2)
        model = lstm.model
        x = np.random.random((1, 120, 7))
        tmp = model(x)
        model.load_weights("tmp/allen_lstm_binary.h5")
        self.model = model
        # Create buffer
        self.buffer = []
        self.buffer_size = buffer_size

    def add_data_point(self, data_point):
        # Add data points to the buffer
        self.buffer.append(data_point)
        # If the buffer size reaches buffer_size, the prediction is made
        if len(self.buffer) == self.buffer_size:
            self.predict()
            # Empty the buffer or keep the buffer size to buffer_size by removing the earliest data points
            self.buffer.pop(0)

    def predict(self, data):
        # Converts the buffer to a numpy array of the shape (buffer_size, 7)
        #input_data = np.array(self.buffer)
        input_data = np.array(data)
        # Convert to model input shape (1, buffer_size, 7)
        input_data = np.expand_dims(input_data, axis=0)
        # Make a prediction
        predictions = self.model.predict(input_data)
        # Processing prediction results
        print("------")
        print("preds: ", predictions)
        print("------")
        #print(predictions)
        return predictions #np.argmax(predictions, axis=0)

    def run(self, data_gen):
        for data_point in data_gen:
            self.add_data_point(data_point)

