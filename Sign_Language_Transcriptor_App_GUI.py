import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Dropout
from PIL import Image, ImageTk
import os
import keyboard
import tkinter as tk
from tkinter import Toplevel, Menu, Label, Button
import webbrowser

# Custom Dropout layer to ensure proper behavior during model loading
class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)

    # Ensures dropout works during both training and inference
    def call(self, inputs, training=None):
        return super().call(inputs, training=training)

# Load the trained model for ASL gesture classification
base_path = os.path.dirname(os.path.abspath(__file__))  # Path of the current file
model_path = os.path.join(base_path, 'efficientnet_hand_gesture_model.h5')  # Model file path
model = load_model(model_path, custom_objects={'swish': swish, 'FixedDropout': FixedDropout})  # Load the model with custom layers

# Preprocesses an image before passing it to the model
def preprocess_image(image):
    # Converts grayscale to RGB if needed and resizes image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = np.array(image, dtype="float32")  # Convert to float32
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Takes a preprocessed image and returns the predicted ASL sign index
def predict_asl_sign(image):
    preprocessed_image = preprocess_image(image)  # Preprocess the input
    prediction = model.predict(preprocessed_image)  # Get model prediction
    predicted_class = np.argmax(prediction, axis=1)  # Get index of highest confidence
    return predicted_class

# Converts the predicted index into its corresponding ASL letter or action
def convert_index_to_sign(index):
    # Maps predicted index to ASL signs, including DEL, SPACE, etc.
    sign_map = {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 
        7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 
        14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 
        20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 
        26: "DEL", 27: "NOTHING", 28: "SPACE"
    }
    sign = sign_map.get(index, "Unknown")  # Return the corresponding sign

    # Simulates keyboard actions based on the sign (e.g., SPACE, DEL)
    if sign == "SPACE":
        keyboard.press_and_release('space')
        return " "
    elif sign == "DEL":
        keyboard.press_and_release('backspace')
        return ""
    
    return sign  # Returns the ASL letter

# Main GUI application class
class ASLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ASL Transcription App")  # Title of the window

        # Set up the window size to match the screen resolution
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}+0+0")  # Set window size to full screen

        self.video_source = 0  # Default to using the primary webcam
        self.vid = cv2.VideoCapture(self.video_source)  # Open webcam video stream

        # Load and set the window icon
        icon_path = os.path.join(base_path, 'assets', 'asl_logo.png')  # Icon file path
        icon_image = Image.open(icon_path)
        icon_image = icon_image.resize((32, 32))  # Resize the icon
        self.icon_photo = ImageTk.PhotoImage(icon_image)  # Convert to PhotoImage format
        self.iconphoto(False, self.icon_photo)  # Set the window icon

        # Header label for the app title
        self.h1_label = tk.Label(self, text="American Sign Language Transcription Using ML", font=("Helvetica", 26, 'bold'))
        self.h1_label.pack(pady=20)  # Padding for spacing

        # Subheader describing the technology used
        self.h2_label = tk.Label(self, text="Made using EfficientNetB0 and custom detection heads", font=("Helvetica", 20))
        self.h2_label.pack(pady=10)

        # Add GitHub repository link for the project
        self.github_link = tk.Label(self, text="Github code repo", font=("Helvetica", 14, 'underline'), fg="blue", cursor="hand2")
        self.github_link.pack(pady=5)
        self.github_link.bind("<Button-1>", lambda e: webbrowser.open_new("https://github.com/RamakrishnaReddyPalle/Sign_Language_Live_Transcription_GUI"))  # Open link when clicked

        # Add a watermark at the bottom-right corner of the window
        self.watermark_label = tk.Label(self, text="Made by: Ram and Vani Nigam", font=("Helvetica", 10))
        self.watermark_label.place(relx=1.0, rely=1.0, anchor='se')  # Position at bottom-right
        self.watermark_label.bind("<Button-1>", lambda e: webbrowser.open_new("https://www.linkedin.com/in/p-rama-krishna-reddy-038b30246/"))  # Left-click for Ram's profile
        self.watermark_label.bind("<Button-3>", lambda e: webbrowser.open_new("https://www.linkedin.com/in/vani-nigam-0707vn/"))  # Right-click for Vani's profile

        # Create a frame to hold the video feed
        self.video_frame = tk.Frame(self) 
        self.video_frame.pack(pady=10)

        self.video_label = Label(self.video_frame)  # Label to display the video feed
        self.video_label.pack()

        # Capture button to take an image and make a prediction
        self.capture_button = Button(self, text="Capture", command=self.capture_image, font=("Helvetica", 12))
        self.capture_button.pack(pady=10)

        # Label to show the transcribed sentence dynamically
        self.sentence_label = Label(self, text="", font=("Arial", 16))
        self.sentence_label.pack(pady=10)

        self.predicted_sentence = ""  # Stores the sentence being transcribed

        # Create the menu bar for additional app options
        menubar = Menu(self)  
        self.config(menu=menubar)

        # Add an info option with examples of ASL signs
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Info", menu=help_menu)
        help_menu.add_command(label="Sign language examples", command=self.show_info)

        self.update()  # Start video stream update loop

    # Updates the video feed from the webcam
    def update(self):
        ret, frame = self.vid.read()  # Read the current frame from the video stream
        if ret:
            self.current_frame = frame  # Save the current frame for prediction
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
            img = Image.fromarray(cv2image)  # Convert to PIL Image
            imgtk = ImageTk.PhotoImage(image=img)  # Convert to ImageTk format for GUI
            self.video_label.imgtk = imgtk  # Update video label with the new frame
            self.video_label.configure(image=imgtk)  # Refresh the displayed image
        self.after(10, self.update)  # Continuously update every 10ms

    # Captures the current video frame and makes a prediction
    def capture_image(self):
        predicted_class = predict_asl_sign(self.current_frame)  # Predict ASL sign for the current frame
        sign = convert_index_to_sign(predicted_class[0])  # Convert predicted index to the actual sign
        self.predicted_sentence += sign  # Append the sign to the ongoing sentence
        self.sentence_label.configure(text=f"Transcribed Sentence: {self.predicted_sentence}")  # Update sentence label

    # Displays additional information in a popup window
    def show_info(self):
        info_window = Toplevel(self)  # Create a new popup window
        info_window.title("Sign language examples")

        # Explanation label in the info window
        info_label = Label(info_window, text="Instructions for Using ASL Transcription App", font=("Helvetica", 14))
        info_label.pack(pady=10)

        # Display ASL sign images as examples
        for letter in ["A", "B", "C"]:
            image_path = os.path.join(base_path, 'assets', f"{letter}.png")  # Path for each ASL letter image
            img = Image.open(image_path)
            img = img.resize((200, 200))  # Resize for uniform display
            imgtk = ImageTk.PhotoImage(img)  # Convert to ImageTk format
            img_label = Label(info_window, image=imgtk)  # Create a label to display the image
            img_label.image = imgtk  # Keep reference to avoid garbage collection
            img_label.pack(pady=5)

# Initialize and run the GUI application
if __name__ == "__main__":
    app = ASLApp()
    app.mainloop()
