🧠 AI Digit Classification Canvas

An interactive Air Canvas based digit classification system powered by a Neural Network trained on the MNIST dataset.
The project allows users to draw digits in the air using hand gestures, processes the captured drawing into MNIST-compatible format, and predicts the digit in real time using a trained neural network model.

This project combines Computer Vision + Deep Learning + Human Interaction to demonstrate how ML models can be integrated with real-world input systems.

🚀 Features

✋ Air Canvas Drawing
Draw digits in the air using hand gestures
Uses computer vision hand tracking to detect finger movement
Creates a virtual canvas to capture the drawn digit

🧠 MNIST Digit Classification
Neural network trained on the MNIST handwritten digit dataset
Classifies digits from 0–9
Predicts the digit drawn on the air canvas

🖼 Image Preprocessing Pipeline
The captured canvas image is automatically processed to match MNIST dataset format:
Background cleaning
Cropping the digit region
Centering the digit
Resizing to 28×28 pixels
Grayscale conversion
Pixel normalization
Reshaping to neural network input format
This ensures the input closely resembles MNIST training images, improving prediction accuracy.

⚡ Real-time Prediction
Processes the drawn digit instantly
Runs inference using the trained neural network
Displays predicted digit directly on screen

🎮 Interactive User Experience
Natural drawing using hand gestures
No mouse, stylus, or touchscreen required
Demonstrates real-time integration between vision systems and AI models

🏗️ Tech Stack
Python
OpenCV
NumPy
pytorch
MediaPipe (for hand tracking)

⚙️ How It Works
The webcam captures hand movement.
Hand tracking detects the index finger.
Finger movement is mapped onto a virtual drawing canvas.

When drawing is completed:
The canvas image is extracted.
Image preprocessing converts it to MNIST format (28×28 grayscale).
The processed image is passed into the trained neural network.
The model predicts the digit and displays the result.

🎯 Applications
Gesture-based interfaces
Educational AI demos
testing MNIST model limitations using non-class numbers

📊 Model

Dataset: MNIST Handwritten Digits
Architecture: Simple Neural Network 
Input: 28×28 grayscale image
Output: Digit classification (0–9)

💡 Future Improvements
CNN model for higher accuracy
good looking UI
Web-based interface
Multi-digit recognition

👨‍💻 Author
Sayar Vaishnav
AI / ML Engineering Student
Interested in Deep Learning, Robotics, and Computer Vision
