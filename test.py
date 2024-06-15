import tensorflow as tf
import numpy as np
from tmp.models import LSTM

print(tf.__version__)

x = np.random.random((1, 120, 7))

lstm = LSTM(n_classes=2)
lstm.compile_model()
model = lstm.model
model(x)

model.load_weights("tmp/allen_lstm_binary.h5")

print(model.summary())

print(model(x))