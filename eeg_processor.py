"""
This is a class that performs real-time EEG processing

raw EEG chunk --(LSL)--> notch filter ----> band pass filter ----> PSD (Max wave return)

Key parameters:
fs: sampling rate - 125Hz
notch_freq: target frequency for notch filter - 50Hz
quality_factor: bandwidths - 30 (Handling a frequency range) / 2 (Handles dozens of frequency ranges)

Other information:
EEG chunk: from LSL, max_samples=self.fs means will receive 125 data per channel at once, leading to a 1-second interval
Data structure: data = np.array(chunk), and the structure of "data" parameter as follows:
           (no column header) f3 f4 .... bip x y z
               (data row 1)   .. .. .... ..  . . .
               (data row 2)   .. .. .... ..  . . .
                ... ...       .. .. .... ..  . . .
             (data row 125)   .. .. .... ..  . . .

    The shape of data is: 11 x 125, and first 7 channels are EEG.
    However, due to the bluetooth ability, some data points may lose
"""

import numpy as np
from pylsl import StreamInlet, resolve_stream
from scipy.signal import butter, lfilter, welch, iirnotch, find_peaks
import matplotlib.pyplot as plt


class EEGProcessor:
    def __init__(self, fs=125, notch_freq=50, quality_factor=30):
        self.fs = fs
        self.notch_freq = notch_freq
        self.quality_factor = quality_factor
        self.b_notch, self.a_notch = self.design_notch_filter()
        self.setup_plot()

    def design_notch_filter(self):
        return iirnotch(self.notch_freq / (self.fs / 2), self.quality_factor)

    def bandpass_filter(self, data, lowcut, highcut, order=5):
        nyq = 0.5 * self.fs
        low = lowcut / nyq # High pass
        high = highcut / nyq # Low pass
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def setup_plot(self):
        """
        This function is for plot the real-time PSD figure
        :return: none
        """
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-', label='Raw PSD')
        self.nms_line, = self.ax.plot([], [], 'ro', label='NMS Peaks')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Power Spectral Density (μV^2/Hz)')
        self.ax.legend()
        self.ax.set_xlim([0, 50])
        self.ax.set_ylim([-100, 100])

    def update_plot(self, freqs, psd, selected_freqs, selected_psd):
        """
        This function is for updating the PSD figure
        :param freqs:
        :param psd:
        :param selected_freqs:
        :param selected_psd:
        :return: none
        """
        #psd[psd < 10] = 0
        #selected_psd[selected_psd < 20] = 0

        self.line.set_data(freqs, psd)
        self.nms_line.set_data(selected_freqs, selected_psd)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        plt.pause(0.1)

    def get_top_peaks(self, psd, freqs, selected_peaks):
        selected_psd = psd[selected_peaks]
        sorted_indices = np.argsort(selected_psd)[-2:]
        top_peaks = selected_psd[sorted_indices]
        top_freqs = freqs[selected_peaks][sorted_indices]
        return top_peaks, top_freqs

    def non_maximum_suppression(self, psd, window_size=5):
        peaks, _ = find_peaks(psd)
        selected_peaks = []
        for peak in peaks:
            start = max(0, peak - window_size)
            end = min(len(psd), peak + window_size)
            if psd[peak] == np.max(psd[start:end]):
                selected_peaks.append(peak)
        return selected_peaks

    def process_data(self):
        print("Looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])
        flag = False
        try:
            while True:
                chunk, timestamps = inlet.pull_chunk(max_samples=self.fs, timeout=1.0)
                if timestamps:  # if there is data coming in
                    data = np.array(chunk)
                    # Notch filter
                    data_filtered = lfilter(self.b_notch, self.a_notch, data[:, 1])  # All EEG data
                    # Band pass filter
                    data_bandpassed = self.bandpass_filter(data_filtered, 5, 40)
                    # PSD
                    freqs, psd = welch(data_bandpassed, self.fs, nperseg=self.fs)
                    # NMS to reduce the non-peak's value
                    selected_peaks = self.non_maximum_suppression(psd)
                    selected_freqs = freqs[selected_peaks]
                    selected_psd = psd[selected_peaks]
                    # Get the local maximum
                    top_peaks, top_freqs = self.get_top_peaks(psd, freqs, selected_peaks)
                    # Edit here based on the finding
                    if len(selected_peaks) >= 2:
                        yield top_peaks[0], top_freqs[0], top_peaks[1], top_freqs[1]
                    else:
                        yield top_peaks[0], top_freqs[0], None, None
                    # Update plot
                    self.update_plot(freqs, psd, selected_freqs, selected_psd)
                    flag = True
                else:
                    if flag:
                        print("Waiting for data ...")
                        flag = False

        except KeyboardInterrupt:
            print("Program stopped manually.")
        finally:
            plt.ioff()

