"""
EEG Drone competition - BCI workshop 2024
University Technology of Sydney
2024-06-07

Real-time EEG --> Machine learning (Need to be implemented) --(return result)--> Tello control

1. Real-time EEG: Using LSL to receive EEG data, and preprocess it. (Output: either processed_EEG or features)
2. Machine Learning: (Input: either processed_EEG or features) - (Output: classification result)
3. Drone Control: (Input: classification result) - perform the corresponding commands.

STRUCTURE:

for ( either processed_EEG or features ) in eeg_processor.process_data():

    Your code for machine learning

    if ( machine learning result ):
        controller.(commands)

except Exception as e:
    controller.emergency_land()
finally:
    controller.land()

"""

import subprocess
import sys


def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(package)


required_packages = ['pylsl', 'djitellopy', 'winsound', 'pygame', 'tensorflow']
for package in required_packages:
    install_and_import(package)

from eeg_app import EEGApp
from ssvep_handler import SSVEPHandler
from tello_controller import TelloController
import tkinter as tk
import threading
from predictor import Predictor


class MainApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.app = EEGApp(self.root, self.handle_data, self.reset)
        self.root.bind('<space>', lambda _: self.start(self))
        self.debug = False
        self.safe_mode = False
        self.lower = 5
        self.upper = 50

        self.controller = TelloController()
        self.stage = 0
        self.distance = 0
        self.is_start = False


    def setup_safety_mode(self):
        self.root.bind('<m>', lambda _: self.switch_safe_mode(self))

        self.root.bind('<w>', lambda _: self.controller.takeoff() if self.safe_mode else print("Take off ..."))
        self.root.bind('<s>', lambda _: self.controller.land() if self.safe_mode else print("Land ..."))
        self.root.bind('<a>', lambda _: self.controller.rotate_desc(90) if self.safe_mode else print("Rotate -90 ..."))
        self.root.bind('<d>', lambda _: self.controller.rotate(90) if self.safe_mode else print("Rotate 90 ..."))

        self.root.bind('<Up>', lambda _: self.controller.move_forward(20) if self.safe_mode else print("Forward 20 .."))
        self.root.bind('<Down>', lambda _: self.controller.move_back(20) if self.safe_mode else print("Back 20 ..."))
        self.root.bind('<Left>', lambda _: self.controller.move_left(20) if self.safe_mode else print("Left 20 ..."))
        self.root.bind('<Right>', lambda _: self.controller.move_right(20) if self.safe_mode else print("Right 20 ..."))

        self.root.bind('<f>', lambda _: self.controller.flip_left() if self.safe_mode else print("Flip ..."))
        self.root.bind('<q>', lambda _: self.controller.move_up() if self.safe_mode else print("Move up ..."))
        self.root.bind('<e>', lambda _: self.controller.move_down() if self.safe_mode else print("Move Down ..."))

    def start(self, event):
        if not self.is_start:
            print("Timer Start!")
            self.is_start = True
            self.app.toggle_timer()

    def end(self):
        if self.is_start:
            print("Timer End!")
            self.is_start = False
            self.app.toggle_timer()
    def switch_safe_mode(self, event):
        self.safe_mode = not self.safe_mode
        print("Sate Mode:", self.safe_mode)

    def reset(self, mode_index):
        print("Reset")
        self.stage = 0
        self.distance = 0
        self.app.reset_timer()
        self.end()

    def handle_data(self, debug, lower, upper):
        self.debug = debug
        self.lower = lower
        self.upper = upper
        print("Received data:")
        print("Debug Mode:", self.debug)
        print("Lower Cutoff Hz:", self.lower)
        print("Upper Cutoff Hz:", self.upper)

        if not self.debug:
            self.setup_safety_mode()

        threading.Thread(target=self.processing_data).start()
        threading.Thread(target=self.processing_ssvep).start()
        self.controller = TelloController(debug=self.debug)

    def processing_ssvep(self):
        from ssvep_stimulus import SSVEPStimulus
        new_window = tk.Toplevel(self.root)
        new_window.title("SSVEP Settings")
        SSVEPStimulus(new_window)
        self.root.focus_set()
    def processing_data(self):
        # Real-time EEG Stream
        processor = SSVEPHandler(self.lower, self.upper)
        predictor = Predictor()
        try:
            if processor.safe_mode == "off":
                for data in processor.process_data():
                    # command = np.argmax(predictor.predict(data), axis=-1)
                    prediction = predictor.predict(data)
                    if prediction[0, 0] > prediction[0, 1] and prediction[0, 0] > 0.99:
                        command = 0
                    elif prediction[0, 0] < prediction[0, 1] and prediction[0, 1] > 0.99:
                        command = 1
                    else:
                        command = 1 # -1
                    if not self.is_start:
                        command = -2
                    print("Prediction Result:", predictor.predict(data))
                    print("Command:", command)
                    if command is not None:
                        # Processing prediction results
                        if processor.safe_mode == "on":
                            command = -1  # enable safe mode
                        if command == 1:
                            if self.stage == 0:
                                print("takeoff")
                                if not self.debug:
                                    self.controller.takeoff()
                                self.stage = 1
                            elif self.stage == 1:
                                if self.distance < 160:
                                    if not self.debug:
                                        self.controller.move_forward(20)
                                    self.distance += 20
                                    print("move 20 cm")
                                else:
                                    self.stage = 2
                            elif self.stage == 2:
                                print("land")
                                if not self.debug:
                                    self.controller.land()
                                self.stage = 3
                                self.end()
            else:
                print("safe mode on")
        except Exception as e:
            print(e)
            # self.controller.emergency()
            self.controller.emergency_land()
        finally:
            print("end")
            self.controller.land()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    main_app = MainApplication()
    main_app.run()
