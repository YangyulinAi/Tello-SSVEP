import emd
import mne
from mne.datasets.sleep_physionet.age import fetch_data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import repeat

from utils import *
from dsp import run_hht, std_scale
from features import *
from models import *
from augmentations import *

plt.close('all')
tf.random.set_seed(42)
np.random.seed(42)

def apply_augs(data):
    data_augs = jitter(data)
    data_augs = scaling(data_augs)
    return data_augs

# Load some data
ALICE, BOB = 0, 1

[alice_files, bob_files] = fetch_data(path='/scratch/rqchia/', 
                                      subjects=[ALICE, BOB], recording=[1])

raw_train = mne.io.read_raw_edf(
    alice_files[0], stim_channel="Event marker", infer_types=True, preload=True
)

# Get the sample frequency from dataset, pre-set in config.py otherwise
time = raw_train.times
fs = np.round(np.mean(1/np.diff(time))).astype(int)
FS = fs

annot_train = mne.read_annotations(alice_files[1])

raw_train.set_annotations(annot_train, emit_warning=False)

annotation_desc_2_event_id = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

# keep last 30-min wake events before sleep and first 30-min wake events after
# sleep and redefine annotations on raw data
annot_train.crop(annot_train[1]["onset"] - 30 * 60, annot_train[-2]["onset"] + 30 * 60)
raw_train.set_annotations(annot_train, emit_warning=False)

window_size = 30
window_shift = 30

events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=window_size
)

# create a new event_id that unifies stages 3 and 4
event_id = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3/4": 3,
    "Sleep stage R": 4,
}

# # Ignore, removes irrelevant channels
# if 'ssvep' not in sample_data_raw_file.name:
#     channels = raw.ch_names
#     drop_channels = [ch for ch in channels if 'EEG' not in ch]
#     raw.drop_channels(drop_channels)
tmax = 30.0 - 1.0 / raw_train.info["sfreq"]  # tmax in included

epochs_train = mne.Epochs(
    raw=raw_train,
    events=events_train,
    event_id=event_id,
    tmin=0.0,
    tmax=tmax,
    baseline=None,
)

example_data = epochs_train.get_data()[:, :2, :]
# channels_last
example_data = np.swapaxes(epochs_train, 1, 2)
example_data = np.vstack(example_data)

# For Hilbert-Huang Transform
hht = []
f = []
freq_edges, freq_centres = emd.spectra.define_hist_bins(0, 5, 5*50, 'linear')

data = example_data.copy()

# channels last
ch_norm = std_scale(data[:, 0]).T
ch_wins = get_windows(ch_norm, int(window_size*FS), int(window_shift*FS))

nsamples = 200
random_samples = np.random.randint(len(ch_wins), size=nsamples)

x = ch_wins[random_samples]
y = events_train.copy()[random_samples, -1]

x_train, x_test, y_train, y_test = split_dataset(x, y)
print(y_train)
print(y_test)

# apply augmentations
x_train_aug = apply_augs(x_train)

tmp_dict = []
m_func = partial(return_all_features, ch=0)

with Pool(cpu_count()) as p:
    tmp_dict_train = p.map(m_func, x_train_aug)
    hht_train = p.starmap(run_hht, zip(x_train_aug, repeat(freq_edges)))

    tmp_dict_test = p.map(m_func, x_test)
    hht_test = p.starmap(run_hht, zip(x_test, repeat(freq_edges)))

f, hht_train_ = zip(*hht_train)
f = f[0]
hht_train = np.array(hht_train_)

_, hht_test_ = zip(*hht_test)
hht_test = np.array(hht_test_)

df_train = pd.DataFrame(tmp_dict_train)
df_test = pd.DataFrame(tmp_dict_test)
print(df_train)

plot_channel_transform(time[:int(window_size*FS)], f, hht_test[10])

lda = get_model('lda')
fnn = get_model('fnn')

fnn.compile_model()
model = fnn.model

# x = df.values[:, :5].astype('float32')
x_feats_train = df_train.values.astype('float32')
x_feats_test = df_test.values.astype('float32')

N_CLASSES = len(np.unique(y))

y_ohe, ohe = do_one_hot(y.reshape(-1, 1))

y_train_ohe = ohe.transform(y_train.reshape(-1,1)).toarray().astype(int)
y_test_ohe = ohe.transform(y_test.reshape(-1,1)).toarray().astype(int)

# model.fit(x_train, do_one_hot(y_train.reshape(-1,1)).toarray())
fnn.fit_model(x_train_aug, y_train_ohe, validation_split=0.2)
y_hat = model.predict(x_test)

plt.figure()
plot_roc_curve(y_test_ohe, y_hat)
pred = np.argmax(y_hat, axis=1)
plt.figure()
plot_confusion_matrix(y_test, pred, labels=np.arange(N_CLASSES))
show_classifier_performance(y_test, pred)

lda.fit(x_feats_train, y_train)
y_hat = lda.predict_proba(x_feats_test)

plt.figure()
plot_roc_curve(y_test_ohe, y_hat)
pred = np.argmax(y_hat, axis=1)
plt.figure()
plot_confusion_matrix(y_test, pred, labels=np.arange(N_CLASSES))
show_classifier_performance(y_test, pred)

plt.show()
