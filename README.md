<p align="left">
  <img src="assets/asl_logo.png" alt="ASL Icon" width="140"/>
</p>

---

# American Sign Language Classification and Live Transcription GUI using EfficientNet, OpenCV, and Tkinter

## Overview

This project is an end-to-end solution for classifying American Sign Language (ASL) gestures and providing real-time transcription. It utilizes **EfficientNet** for gesture classification and employs **OpenCV** for live video capture, and **Tkinter** to create a GUI-based transcription system.

The system was trained on an ASL dataset from Kaggle and provides a real-time transcription interface using the computer’s webcam. The model classifies 29 ASL signs, including all the alphabet letters and special signs like "SPACE", "DELETE", and "NOTHING".

---

## Dataset

The dataset used for training this model is from [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). It contains images of hand gestures representing the letters of the alphabet along with a few special symbols like "SPACE", "DELETE", and "NOTHING". 

- **Training Set**: 87,000 images (3,000 images per class for 29 classes)
- **Test Set**: 29,000 images (1,000 images per class for 29 classes)

Each image is resized to 224x224 pixels, consistent with EfficientNet's input size.

---

## Model Architecture

The model uses **EfficientNetB0**, a pre-trained model on ImageNet, as the base for transfer learning. The base layers are frozen initially, and a custom classifier is added on top. Here are the key modifications and components:

- **Global Average Pooling** for dimensionality reduction after the EfficientNet base layers.
- **Dense Layer** with 128 units and **L2 regularization** to prevent overfitting.
- **Dropout Layers** to further prevent overfitting.
- **Final Dense Layer** with 29 output nodes for the ASL classes using **softmax** activation.

**Model Summary**:

```text
Input: (224, 224, 3)
EfficientNetB0 base model (ImageNet weights, frozen layers initially)
GlobalAveragePooling2D
Dropout(0.5)
Dense(128 units, L2 regularization)
Dropout(0.4)
Dense(29 units, softmax)
```

### Training Strategies

1. **Transfer Learning**: The base layers from EfficientNetB0 were frozen initially. After training the top layers, the last 20 layers of the base model were unfrozen for fine-tuning.
2. **Data Augmentation**: To avoid overfitting and improve generalization, the following augmentations were applied:
   - Random rotations, zoom, brightness adjustments, and horizontal flips.
3. **Early Stopping and Model Checkpointing**: 
   - Early stopping was implemented with a patience of 7 epochs to prevent overfitting.
   - Model checkpointing saved the best model during training.
   
4. **Optimizer**: Adam optimizer with an initial learning rate for transfer learning, which was reduced during fine-tuning.

### Training Results

- **Test Accuracy**: 93.74%
- **Final Model File**: `efficientnet_hand_gesture_model.h5`

---

## Data Preprocessing

Each image in the dataset was preprocessed with the following steps:

1. **Resizing**: Images were resized to 224x224 pixels.
2. **Normalization**: Pixel values were normalized to the range [0, 1].
3. **Augmentation**: For training, random augmentations (rotation, brightness, flip) were applied.

---

## Running the Model

### Clone and Setup

To get started, clone the repository and install the dependencies.

```bash
git clone <https://github.com/RamakrishnaReddyPalle/Sign_Language_Live_Transcription_GUI.git>
```
```bash
cd path\to\cloned\folder
```
```bash
pip install -r requirements.txt
```

Ensure that the dataset is downloaded and placed in the appropriate directories (e.g., `dataset_asl/asl_alphabet_train/` for training and `dataset_asl/asl_alphabet_test/` for testing).

### Training the Model

To train your own custom version of my model, execute the `ASL_training_pipeline.ipynb` script.
This script will train the model using the ASL dataset and save the best version of the model to `efficientnet_hand_gesture_model.h5`.

### Inference and Transcription

The `inference_pipeline.ipynb` script can be used for predicting ASL signs from live webcam input.
This will capture frames from your webcam, classify them, and output the predicted ASL sign.

### Running the GUI

To launch the transcription GUI, execute the following command:

```bash
python Sign_Language_Transcriptor_App_GUI.py
```

This opens a window where live video from your webcam is displayed. You can start predicting ASL signs by pressing specific keys:

- **'p'**: Predict the ASL sign in the current frame.
- **'q'**: Quit the application.

The GUI supports live transcription directly into text using special ASL signs for space and delete.

---

## GUI Features

The graphical interface is built using **Tkinter** for ease of use and integration with OpenCV. Here are some highlights:

- **Live Webcam Input**: Displays a real-time video feed.
- **Real-time Sign Prediction**: Classifies hand gestures in real-time, printing the result to the terminal.
- **Keyboard Integration**: Supports automatic typing of letters, spaces, and backspace using `keyboard` module.
- **Easy-to-use Controls**: Simple key bindings for starting predictions (`p` key) and exiting the program (`q` key).

##  Demos
![GUI Screenshot](assets/Screenshot%202024-09-29%20035526.png)
![GUI Screenshot](assets/Screenshot%202024-09-29%20035611.png)
### *Few screenshots of our ASL Transcription GUI in action.*
---

## Results

The model achieved decent accuracy (93.74%) on the test set and performs well in real-time transcription via the GUI.

- **Augmentation** and **Regularization** helped in improving the generalization of the model.
- Fine-tuning further enhanced the model performance by unfreezing the last layers of the EfficientNetB0 base model.

---

## Future Work

- **Model Optimization**: The model can be optimized further by experimenting with different EfficientNet variants (B1, B2, etc.) and hyperparameter tuning.
- **Extending the Dataset**: Including more ASL signs or gestures for dynamic signs (e.g., words or phrases) can be considered.

---

## Conclusion

This project demonstrates how deep learning models can be utilized for American Sign Language gesture classification and transcription. By leveraging EfficientNet, OpenCV, and Tkinter, we developed a robust system that can classify 29 different ASL gestures in real-time.

---

## Contributors

- [Ramakrishna Reddy Palle](https://www.linkedin.com/in/p-rama-krishna-reddy-038b30246/)
- [Vani Nigam](https://www.linkedin.com/in/vani-nigam-0707vn/) 

---

Feel free to contribute or report issues in the repository.

---

This README gives an overall explanation of the key concepts, methods, and results of your project, including sections for dataset, model architecture, training strategies, GUI, and inference. Let me know if you need adjustments or further details!
