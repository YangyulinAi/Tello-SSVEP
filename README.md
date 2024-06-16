# SSVEP-Based Drone Control System

## Introduction
This repository contains the source code for a drone control system utilizing the Steady-State Visual Evoked Potential (SSVEP) paradigm. The system is designed to interpret user intent through SSVEP signals and translate it into drone control commands. It incorporates a trained machine learning model for signal classification.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow
- Required Python packages: `numpy`, `pandas`, `matplotlib`, `scipy` etc. (Auto-installed)

### Installation
1. Clone the repository to your local machine: https://github.com/YangyulinAi/Tello-SSVEP.git
2. Navigate to the cloned directory


### Setup
1. Replace the default model weight file with the trained model:
- Remove the existing `model.h5` file in the `model` directory.
- Rename your trained model weight file to `model.h5` and place it in the `model` directory.

### Usage
1. Ensure that the X.on hardware connection is established properly.
2. Run the main application using Python: python main.py
3. A main menu will appear with options to start training, testing, or control the drone using the SSVEP interface.

#### Main Menu
![image](https://github.com/YangyulinAi/Tello-SSVEP/assets/78721047/63d85bb8-ef2e-4cfc-a372-0fa816901199)

 - **Debug Mode:** The default option, On, allows you to simulate drone operation through the console output. Off, connect the drone for a real test.
 - **Lower Cutoff Hz:** Pre-processing parameter for band-pass filter.
 - **Upper Cutoff Hz:** Pre-processing parameter for band-pass filter.
 - **Submit Button:** Click to start.

#### SSVEP Setup
![image](https://github.com/YangyulinAi/Tello-SSVEP/assets/78721047/fd8c129e-deab-4c93-a09d-7de315127e60)

- **Number of Stumuli:** Integer 1-4 (Min:1, Max:4)
- **Duration:** Seconds (Min: 1)
- **Frequencies:** Please keep the number of stimuli equal to the number of frequencies, using comma separations with no spaces.
- **Mode:** Train is collecting data (with Lab Recorder); Test is for control the drone in real-time.
- **For BCI Workshop 2024 participants, the recommended parameters here are, Number of Stumuli: 1, Frequencies: 15**

#### Control Panel
![image](https://github.com/YangyulinAi/Tello-SSVEP/assets/78721047/8ca40adf-9175-4f8e-9667-e3021099aadd)

There is no need to do anything here, when ready, click on the control panel window and press space, the timer will start automatically. When the drone lands, the timer will automatically pause. When the drone is moved back to the starting position, it can be reset by pressing reset to resume operation.
The other buttons are for emergency mode operation to keep the drone safe.


## Features
- **Training Mode:** Train the machine learning model with new SSVEP data.
- **Testing Mode:** Real-time control of the drone using the SSVEP signals captured from the user.
  
## Contributing
Contributions to this project are welcome. Please fork the repository and submit pull requests with your enhancements.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Thanks to all the contributors who have invested their time in improving this project.



