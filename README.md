This repository contains a prototype for an Explainable AI (XAI) Medical Diagnostic Dashboard designed to assist in detecting pneumonia from chest X-ray scans. Built as a student academic project, the goal of this application is to bridge the gap between black-box deep learning models and clinical interpretability. By utilizing transfer learning and generating neural activation maps, the system not only predicts the presence of pneumonia but visually explains where the model is looking to make that prediction.

Core Features 
Deep Learning Inference: Utilizes a custom PyTorch model built on the ResNet50 architecture, modified to output a binary classification for Normal vs. Pneumonia.   Explainable AI (Grad-CAM): Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heatmaps that highlight the regions of the lung influencing the AI's decision.   Interactive Clinical Interface: Features a vibrant, glassmorphism-styled Streamlit UI named "RadixAI", complete with dynamic patient demographic inputs, inference metric visualizers, and simulated EMR export capabilities.   Automated Clinical Findings: Dynamically generates clinical impressions, confidence intervals, and suggested next-step protocols based on the AI's primary prediction.   Unit Testing Setup: Includes a pytest suite designed to validate model initialization, ensure correct image tensor shapes, and verify the normalization constraints of the heatmap generations. 

Technology StackThis project relies on the following key libraries:   Deep Learning: torch, torchvisionFrontend Framework: streamlitComputer Vision & Imaging: opencv-python-headless, PillowData Processing: numpy, pandas (for telemetry mapping in the UI)Testing: pytest

Installation & SetupClone the repository:Bashgit clone https://github.com/om3692/RadixAI-Dashboard.git
cd RadixAI-Dashboard
Create a virtual environment (Recommended):Bashpython -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install the dependencies:
Ensure you install the required packages using the provided requirements file.   Bashpip install -r requirements.txt
Run the application:Bashstreamlit run app.py

<img width="602" height="362" alt="image" src="https://github.com/user-attachments/assets/1d10195f-673f-44f8-99f2-19e4eb5f2a68" />
<img width="602" height="361" alt="image" src="https://github.com/user-attachments/assets/b0d19281-91d5-49a8-b184-c5ca5b39d440" />
<img width="940" height="516" alt="image" src="https://github.com/user-attachments/assets/9efa0de5-aa53-440e-bda3-5a0f41360d86" />
<img width="602" height="362" alt="image" src="https://github.com/user-attachments/assets/466c160c-e052-4cd1-936e-fec87020fccf" />
