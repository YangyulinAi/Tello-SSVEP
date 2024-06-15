import tkinter as tk
from tkinter import ttk
import time

class EEGApp:
    def __init__(self, master, callback_ini, callback_mode):
        """Initialize the EEG application with the main window."""
        self.master = master
        self.callback = callback_ini  # Store the callback function
        self.callback_mode = callback_mode
        self.master.title("EEG Data Input")

        # Initialize timer
        self.is_running = False
        self.start_time = 0
        self.elapsed_time = 0

        # Initialize variables
        self.debug_modes = ["On", "Off"]
        self.debug_mode_var = tk.StringVar(value="On")

        self.lower_cutoff_var = tk.DoubleVar(value=5)
        self.upper_cutoff_var = tk.DoubleVar(value=45)

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        """Create widgets for the GUI."""
        # Debug mode selection
        debug_mode_label = ttk.Label(self.master, text="Debug Mode:")
        debug_mode_label.grid(column=0, row=0, padx=10, pady=10)
        debug_mode_combo = ttk.Combobox(self.master, textvariable=self.debug_mode_var, values=self.debug_modes)
        debug_mode_combo.grid(column=1, row=0, padx=10, pady=10)

        # Lower cutoff frequency
        lower_cutoff_label = ttk.Label(self.master, text="Lower Cutoff Hz:")
        lower_cutoff_label.grid(column=0, row=1, padx=10, pady=10)
        lower_cutoff_entry = ttk.Entry(self.master, textvariable=self.lower_cutoff_var)
        lower_cutoff_entry.grid(column=1, row=1, padx=10, pady=10)

        # Upper cutoff frequency
        upper_cutoff_label = ttk.Label(self.master, text="Upper Cutoff Hz:")
        upper_cutoff_label.grid(column=0, row=2, padx=10, pady=10)
        upper_cutoff_entry = ttk.Entry(self.master, textvariable=self.upper_cutoff_var)
        upper_cutoff_entry.grid(column=1, row=2, padx=10, pady=10)

        # Submit button
        submit_button = ttk.Button(self.master, text="Submit", command=self.submit)
        submit_button.grid(column=0, row=3, columnspan=2, padx=10, pady=10)


    def move_window_to_right(self):
        # Calculate new x position to move the window to the right side of the screen
        screen_width = self.master.winfo_screenwidth()
        new_x_position = screen_width - 600
        self.master.geometry(f"+{new_x_position}+50")
    def submit(self):
        """Handle the submit button click event."""
        # Clear the initial window
        for widget in self.master.winfo_children():
            widget.destroy()

        # Create the new window layout
        self.master.title("Response Page")

        # Button on the left
        button = ttk.Button(self.master, text="Reset", command=self.reset)
        button.grid(column=0, row=0, padx=5, pady=20, sticky='w')

        # Label on the right
        self.timer_label = ttk.Label(self.master, text="00:00:000", relief="solid", width=10)
        self.timer_label.grid(column=2, row=0, padx=5, pady=20, sticky='e')

        # Center labels on the window
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=2)
        self.master.grid_columnconfigure(2, weight=1)

        # Directional buttons
        self.create_directional_buttons()

        self.move_window_to_right()

        # Retrieve the input values
        debug_mode = self.debug_mode_var.get()
        lower_cutoff = self.lower_cutoff_var.get()
        upper_cutoff = self.upper_cutoff_var.get()
        print(f"Debug Mode: {debug_mode}, Lower Cutoff: {lower_cutoff} Hz, Upper Cutoff: {upper_cutoff} Hz")
        if debug_mode == "On":
            debug = True
        else:
            debug = False

        # Call the callback function with the data
        self.callback(debug, lower_cutoff, upper_cutoff)

    def toggle_timer(self):
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()
            self.update_timer()
        else:
            self.is_running = False

    def reset_timer(self):
        self.timer_label.config(text="00:00:000")  # Reset timer display when stopped

    def update_timer(self):
        if self.is_running:
            elapsed_time = time.time() - self.start_time
            mins, secs = divmod(int(elapsed_time), 60)
            millis = int((elapsed_time - int(elapsed_time)) * 1000)
            self.timer_label.config(text="{:02d}:{:02d}:{:03d}".format(mins, secs, millis))
            self.master.after(50, self.update_timer)  # Update every 50 ms to simulate millisecond display

    def create_directional_buttons(self):
        button_frame = ttk.Frame(self.master)
        # Make sure the button_frame is centered by setting columnspan to the number of columns used by the labels
        button_frame.grid(column=0, row=1, columnspan=3, pady=20)

        # Buttons
        self.up_button = ttk.Button(button_frame, text="Up")
        self.up_button.grid(column=1, row=0)

        self.left_button = ttk.Button(button_frame, text="Left")
        self.left_button.grid(column=0, row=1)

        self.space_button = ttk.Button(button_frame, text="Space")
        self.space_button.grid(column=1, row=1)

        self.right_button = ttk.Button(button_frame, text="Right")
        self.right_button.grid(column=2, row=1)

        self.down_button = ttk.Button(button_frame, text="Down")
        self.down_button.grid(column=1, row=2)

        # Configure the button frame's grid to distribute space evenly
        for i in range(3):
            button_frame.grid_columnconfigure(i, weight=1)
            button_frame.grid_rowconfigure(i, weight=1)

    def highlight_button(self, button):
        """Highlight a button when a key is pressed."""
        button.state(['pressed'])
        self.master.after(100, lambda: button.state(['!pressed']))

    def reset(self):
        """Activate the button corresponding to the pressed key."""
        self.callback_mode(1)
