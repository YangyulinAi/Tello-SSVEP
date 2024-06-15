import tkinter as tk
import pygame
import threading
import time
from tkinter import ttk
from pylsl import StreamInfo, StreamOutlet
import winsound

class SSVEPStimulus:
    def __init__(self, master):
        # Global creation of LSL stream
        self.master = master
        self.info = StreamInfo('BCIWorkshop2024', 'Markers', 1, 0, 'string', 'BCIWorkshop2024')
        self.outlet = StreamOutlet(self.info)
        self.DEFAULT_FREQUENCIES = [7.0, 8.0, 9.0, 11.0]  # Set default frequencies
        print("LSL Stream has been set up. Waiting for stream activation...")
        time.sleep(2)
        print("LSL Stream is active.")
        self.setup_widgets()

    def setup_widgets(self):
        # Set the number of stimuli
        tk.Label(self.master, text="Enter Number of Stimuli (1-4):").pack()
        self.num_stimuli_entry = tk.Entry(self.master)
        self.num_stimuli_entry.pack()

        # Set the duration
        tk.Label(self.master, text="Enter Duration (seconds):").pack()
        self.duration_entry = tk.Entry(self.master)
        self.duration_entry.pack()

        # Set input frequencies
        tk.Label(self.master, text="Enter Frequencies (Hz, comma-separated, optional):").pack()
        self.frequency_entry = tk.Entry(self.master)
        self.frequency_entry.pack()

        # Mode selection dropdown menu
        mode_label = tk.Label(self.master, text="Select Mode:")
        mode_label.pack()
        self.mode_var = tk.StringVar()
        mode_combobox = ttk.Combobox(self.master, textvariable=self.mode_var, values=["train", "test"])
        mode_combobox.pack()
        mode_combobox.set("train")  # Default set to train

        # Start button
        start_button = tk.Button(self.master, text="Start Stimulus", command=self.open_stimulus_window)
        start_button.pack()

    def open_stimulus_window(self):
        mode = self.mode_var.get()
        frequencies = [float(freq) for freq in self.frequency_entry.get().split(',') if freq]
        duration = float(self.duration_entry.get())
        num_stimuli = int(self.num_stimuli_entry.get())
        threading.Thread(target=self.start_stimulus, args=(frequencies, duration, num_stimuli, mode)).start()

    def start_stimulus(self, frequencies, duration, num_stimuli, mode, close_eyes=False):
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("SSVEP Stimulus")

        black = (0, 0, 0)
        white = (255, 255, 255)
        colors = [black, white]  # Use black and white colors to alternate display

        # Ensure the frequencies list has enough values by appending defaults if needed
        frequencies += self.DEFAULT_FREQUENCIES[len(frequencies):num_stimuli]

        periods = [1.0 / freq for freq in frequencies]  # Calculate periods for each frequency
        on_durations = [period / 2 for period in periods]  # Calculate on durations
        off_durations = [period / 2 for period in periods]  # Calculate off durations

        start_time = time.time()
        change_time = start_time
        current_index = 0  # Current frequency index

        # Set the position of each frequency's rectangle based on index
        rects = []
        if num_stimuli == 1:
            rects = [pygame.Rect(350, 250, 100, 100)]
        elif num_stimuli == 2:
            rects = [pygame.Rect(250, 250, 100, 100), pygame.Rect(450, 250, 100, 100)]
        elif num_stimuli == 3:
            rects = [pygame.Rect(150, 250, 100, 100), pygame.Rect(350, 250, 100, 100), pygame.Rect(550, 250, 100, 100)]
        elif num_stimuli == 4:
            rects = [pygame.Rect(250, 150, 100, 100), pygame.Rect(450, 150, 100, 100),
                     pygame.Rect(250, 350, 100, 100), pygame.Rect(450, 350, 100, 100)]

        running = True
        last_flip_times = [start_time] * num_stimuli
        states = [0] * num_stimuli

        while running and (time.time() - start_time) < duration:
            for event in pygame.event.get():
                if event.type is pygame.QUIT:
                    running = False

            current_time = time.time()

            if mode == "train" and current_time - change_time >= 5:  # Alternate frequencies every five seconds
                current_index = (current_index + 1) % num_stimuli  # Cycle through each frequency
                change_time = current_time
                current_freq = frequencies[current_index]
                self.outlet.push_sample([f"Freq_{current_freq}Hz"])  # Send event marker
                print(f"Freq_{current_freq}Hz")
                if num_stimuli == 1:
                    if close_eyes:
                        self.outlet.push_sample([f"close_eyes"])  # Send event marker
                        print(f"close_eyes")
                        close_eyes = False
                        winsound.Beep(500, 100)

                    else:
                        self.outlet.push_sample([f"open_eyes"])  # Send event marker
                        print(f"open_eyes")
                        close_eyes = True
                        winsound.Beep(1000, 100)

            screen.fill(black)  # Clear the screen to black

            # Decide display logic based on mode
            if mode == "train":
                for i in range(num_stimuli):
                    if i == current_index:
                        if current_time - last_flip_times[i] >= (on_durations[i] if states[i] == 0 else off_durations[i]):
                            states[i] = 1 - states[i]  # Toggle state
                            last_flip_times[i] = current_time
                        pygame.draw.rect(screen, colors[states[i]], rects[i])
            else:  # In "test" mode, display all stimuli simultaneously
                for i in range(num_stimuli):
                    pygame.draw.rect(screen, colors[time.time() % on_durations[i] < on_durations[i] / 2], rects[i])

            pygame.display.flip()

        pygame.quit()

# Create a Tkinter window
root = tk.Tk()
root.title("SSVEP Settings")
app = SSVEPStimulus(root)
root.mainloop()
