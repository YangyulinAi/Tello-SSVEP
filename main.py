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

required_packages = ['pylsl','djitellopy','winsound','pygame','tensorflow']
for package in required_packages:
    install_and_import(package)

from eeg_app import EEGApp
from SSVEP_handler import SSVEPHandler
from tello_controller import TelloController
import tkinter as tk
import threading
from predictor import Predictor
import numpy as np

class MainApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.app = EEGApp(self.root, self.handle_data, self.mode_change)
        self.controller = TelloController(debug=True)

        self.channel = 0
        self.lower = 5
        self.upper = 50

        self.mode = 1

        self.root.bind('<space>', self.toggle_fly)
        self.root.bind('<Up>', lambda _: self.controller.move_forward(20))
        self.root.bind('<Down>', lambda _: self.controller.move_back(20))
        self.root.bind('<Left>', lambda _: self.controller.move_left(20))
        self.root.bind('<Right>', lambda _: self.controller.move_right(20))
        self.root.bind('<f>', lambda _: self.controller.flip_left())
        self.root.bind('<w>', lambda _: self.controller.flip_forward())
        self.root.bind('<s>', lambda _: self.controller.flip_backward())
        self.root.bind('<a>', lambda _: self.controller.flip_left())
        self.root.bind('<d>', lambda _: self.controller.flip_right())
        self.root.bind('<q>', lambda _: self.controller.move_up())
        self.root.bind('<e>', lambda _: self.controller.move_down())
    def toggle_fly(self,event):
        print("Safe mode control ...")
        if self.controller.check_flying_status():
            self.controller.land()
        else:
            self.controller.takeoff()

    def mode_change(self, mode_index):
        print("mode", mode_index)
        self.mode = mode_index
    def handle_data(self, channel, lower, upper):
        self.channel = channel
        self.lower = lower
        self.upper = upper
        print("Received data:")
        print("Channel:", self.channel)
        print("Lower Cutoff Hz:", self.lower)
        print("Upper Cutoff Hz:", self.upper)

        threading.Thread(target=self.processing_data).start()
        threading.Thread(target=self.processing_ssvep).start()


    def processing_ssvep(self):
        from ssvep_stimulus import SSVEPStimulus
        ssvep = SSVEPStimulus()

    def processing_data(self):
        # Real-time EEG Stream
        processor = SSVEPHandler(self.channel, self.lower, self.upper)
        model_path = './allen_lstm_binary_1.keras'
        predictor = Predictor(model_path)
        is_fly = False
        try:

            if processor.safe_mode == "off":
                for data in processor.process_data():
                    print("HERE IS DATA", data.shape)
                    command = np.argmax(predictor.predict(data), axis=-1)
                    print("Prediction Result:", predictor.predict(data))
                    print("Command:", command)
                    if command is not None:
                        # Processing prediction results
                        if processor.safe_mode == "on":
                            command = -1  # enable safe mode
                        #print(command)
                        if command == 1:
                            if not is_fly:
                                print("takeoff")
                                is_fly = True
                                #self.controller.takeoff()
                            else:
                                print("land")
                                is_fly = False # For safety purpose can comment this line
                                #self.controller.land()
                        elif command == 2:
                            if is_fly:
                                print("fly task performed")
                                if self.mode == 1:
                                    #self.controller.preset_command()
                                    print("1")
                                elif self.mode == 2:
                                    #self.controller.preset_command_2()
                                    print("2")
                                else:
                                    #self.controller.move_forward(20)
                                    print("3")
            else:
                print("debugdebug")
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
