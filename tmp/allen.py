from os.path import exists
import ipdb
import emd
import mne
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import repeat

from utils import *
from dsp import run_hht, std_scale, butter_bandpass_filter, movingaverage
from features import *
from models import *
from augmentations import *

plt.close('all')
tf.random.set_seed(42)
np.random.seed(42)

def preprocess(data, fs, do_plot=False):
    # 5 - 40 bandpass
    filt = butter_bandpass_filter(data, 5, 40, fs=fs, axis=0)

    # moving average (4)
    # filt = movingaverage(filt, 4, axis=0)

    # std scale
    # filt = std_scale(filt)

    if do_plot:
        plt.subplot(211)
        plt.plot(data[:, 0])
        plt.plot(filt[:, 0])

        plt.subplot(212)
        plt.plot(*run_fft(data[:,0], fs=fs))
        plt.plot(*run_fft(filt[:,0], fs=fs))

        plt.show()

    return filt

def apply_augs(data):
    data_augs = jitter(data, sigma=0.1)
    # data_augs = time_warp(data_augs)
    data_augs = scaling(data_augs, sigma=0.05)
    # data_augs = permutation(data_augs)
    return data_augs

# FILE = "/data/rqchia/bciworkshop/13_06_Allen_with_11Hz15Hz25Hz35Hz.xdf"
FILE = "/data/rqchia/bciworkshop/14_06_Allen_open_close_eyes_25Hz.xdf"

window_size = 1 # seconds
window_shift = 0.25 # seconds

(data_dict, marker_dict), header = load_xdf(FILE)

fs = get_xdf_fs(data_dict)

ch_names = get_channel_names(data_dict)

marker_labels, marker_timestamps = get_marker_data(marker_dict, re_str='eyes')

data, data_timestamps = get_eeg_data(data_dict)

# segment data
t0_array = marker_timestamps[:-1]
t1_array = marker_timestamps[1:]

keys = np.unique(marker_labels)
data_segments = {int(k): [] for k in keys if not np.isnan(k) and int(k) in
                [0, 1]}
# data_segments = {int(k): [] for k in keys if not np.isnan(k) and int(k) in
#                 [11, 35]}

for lbl, t0, t1 in zip(marker_labels[:-1], t0_array, t1_array):
    mask = (data_timestamps >= t0) & (data_timestamps < t1)

    # get timestamps in period of interest
    data_segment = data[mask]

    data_segment = preprocess(data_segment, fs)

    # create windows from period of interest
    data_wins = get_windows(data_segment, int(window_size*fs),
                            int(window_shift*fs))

    data_segments[int(lbl)].append(data_wins)

n_classes = len(data_segments)

# Create arrays of labels and input data
x, y = [], []
for k, v in data_segments.items():
    x_in = np.vstack(v)
    x.append(x_in)
    y.append(np.repeat(k, len(x_in)))

x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0).reshape(-1, 1)

# take P channels only [5, 6]
x = x[... , :7]
y_ohe, ohe = do_one_hot(y, max_categories=n_classes)
y_ohe = y_ohe.toarray().astype(int)

# Split train, test
x_train, x_test, y_train, y_test = split_dataset(x, y_ohe, stratify=y)

# Augment training data
x_train_aug = apply_augs(x_train)
# x_train_aug = x_train

# extract features
tmp_dict = []
m_func_ch6 = partial(return_all_features, ch=6)
m_func_ch7 = partial(return_all_features, ch=7)

with Pool(cpu_count()) as p:
    tmp_dict_train_ch6 = p.map(m_func_ch6, x_train_aug)
    tmp_dict_train_ch7 = p.map(m_func_ch7, x_train_aug)

    tmp_dict_test_ch6 = p.map(m_func_ch6, x_test)
    tmp_dict_test_ch7 = p.map(m_func_ch7, x_test)

df_train_ch6 = pd.DataFrame(tmp_dict_train_ch6)
df_train_ch7 = pd.DataFrame(tmp_dict_train_ch7)

df_test_ch6 = pd.DataFrame(tmp_dict_test_ch6)
df_test_ch7 = pd.DataFrame(tmp_dict_test_ch7)

df_train = pd.concat([df_train_ch6, df_train_ch7], axis=1)
df_test = pd.concat([df_test_ch6, df_test_ch7], axis=1)

df_train.replace([np.inf, -np.inf], 0, inplace=True)
df_test.replace([np.inf, -np.inf], 0, inplace=True)
df_train.replace([np.nan, -np.nan], 0, inplace=True)
df_test.replace([np.nan, -np.nan], 0, inplace=True)
print(df_train.head())
print(df_train.shape)

# Init neural net
epochs = 50
mdl_str = 'lstm'
my_model = get_model(mdl_str, n_classes=n_classes, verbose=1, epochs=epochs,
                     lr=1e-4)
patience = epochs//10
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=patience),
    # tf.keras.callbacks.EarlyStopping(patience=patience)
]

# Train and validate
my_model.compile_model()
# try:
#     model = load_model(f'./trained_models/allen_{mdl_str}_binary.keras')
# except:
my_model.fit_model(
    x_train_aug, y_train,
    validation_split=0.25,
    callbacks=callbacks,
)
model = my_model.model
save_model(model, f'allen_{mdl_str}_binary')

y_hat = model.predict(x_test)

plt.figure()
plot_roc_curve(y_test, y_hat)
pred = np.argmax(y_hat, axis=1)
plt.figure()
y_test_le = np.where(y_test==1)[1]
plot_confusion_matrix(y_test_le, pred, labels=data_segments.keys())
show_classifier_performance(y_test_le, pred)

plt.show()

# Init LDA
mdl_str = 'svm'
model = get_model(mdl_str, n_classes=n_classes)
x_train = df_train.values.astype('float32')
x_test = df_test.values.astype('float32')

y_train_le = np.where(y_train==1)[1]

model.fit(x_train, y_train_le)

# Evaluate
# y_hat = model.predict_proba(x_test)
y_hat = model._predict_proba_lr(x_test)

plt.figure()
plot_roc_curve(y_test, y_hat)
pred = np.argmax(y_hat, axis=1)
plt.figure()
y_test_le = np.where(y_test==1)[1]
plot_confusion_matrix(y_test_le, pred, labels=data_segments.keys())
show_classifier_performance(y_test_le, pred)

save_model(model, f'allen_{mdl_str}')

plt.show()
