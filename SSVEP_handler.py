import numpy as np
from pylsl import StreamInlet, resolve_stream
from scipy.signal import butter, lfilter, iirnotch
import matplotlib.pyplot as plt


class SSVEPHandler:
    def __init__(self, channel=0, lower=5, upper=50, fs=120, notch_freq=50, quality_factor=30):
        self.fs = fs  # Sampling rate
        self.channel = channel  # Channel
        self.lower = lower  # Lower frequency for bandpass filter
        self.upper = upper  # Upper frequency for bandpass filter
        self.notch_freq = notch_freq  # Notch filter frequency
        self.quality_factor = quality_factor  # Quality factor for notch filter

        # Design notch filter
        self.b_notch, self.a_notch = self.design_notch_filter()

        # Initialize buffer and index
        self.buffer = np.zeros((120, 7))  # Buffer size for 125 samples per second, 7 columns per sample
        self.index = 0  # Current index position

        print("Looking for an EEG stream...")
        self.streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(self.streams[0])
        print("EEG stream receiving...")

        self.safe_mode = "off"

    def design_notch_filter(self):
        # Design notch filter
        return iirnotch(self.notch_freq / (self.fs / 2), self.quality_factor)

    def bandpass_filter(self, data, order=5):
        # Apply bandpass filter
        nyq = 0.5 * self.fs
        low = self.lower / nyq
        high = self.upper / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def process_data(self):
        is_data_coming = False
        try:
            while True:
                chunk, timestamps = self.inlet.pull_chunk(max_samples=self.fs, timeout=1.0)
                if timestamps:  # If there is data coming in
                    data = np.array(chunk)
                    #data_filtered = lfilter(self.b_notch, self.a_notch, data.T)
                    #data_bandpassed = self.bandpass_filter(data_filtered)
                    processed_data = data[:,:7]

                    print("debug",processed_data.shape)


                    yield processed_data
                    is_data_coming = True
                else:
                    if is_data_coming:
                        print("Waiting for LSL Stream ...")
                        is_data_coming = False
        except KeyboardInterrupt:
            print("Program stopped manually.")
        finally:
            plt.ioff()
