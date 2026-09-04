🚦 TrafficGuard AI

🇮🇳 Intelligent Traffic Violation Detection & Road Safety Platform

TrafficGuard AI is a computer-vision web application developed to assist traffic-safety analysis from road-scene images.

The project combines a 25-class YOLO detection model, a specialized helmet model, number-plate detection + OCR, confidence analysis, and a human-review decision path in one dashboard.

AI detects → confidence is evaluated → reliable cases are reported → uncertain cases can be sent for human review.

🎯 What We Built

The system is designed around four target traffic violations:

No Helmet

Triple Riding

Phone While Driving

Seatbelt Violation

The four targets are handled using the two trained traffic models:

main_25class_best.pt — primary 25-class YOLO model; the notebook maps Triple Riding, Phone While Driving, and Seatbelt Violation to classes 3, 4, and 9.

helmet_balanced_best.pt — specialized helmet model; the notebook maps no_helmet to class 1.

Number-plate processing is an additional detection + OCR stage used when a violation is found.

🧠 Complete AI Pipeline

                         TRAFFIC IMAGE
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
      25-CLASS YOLO                       HELMET MODEL
 main_25class_best.pt                helmet_balanced_best.pt
             │                                 │
             ├── Triple Riding                └── No Helmet
             ├── Phone While Driving
             └── Seatbelt Violation
             │
             └───────────────┬─────────────────┘
                             ▼
                  VIOLATION FOUND?
                     /             \
                   NO               YES
                   │                 │
                   ▼                 ▼
             NOT DETECTED      PLATE DETECTOR
                                     │
                                     ▼
                                    OCR
                                     │
                                     ▼
                            CONFIDENCE ANALYSIS
                                     │
                           ┌─────────┼─────────┐
                           ▼         ▼         ▼
                       DETECTED  HUMAN REVIEW
                                           NOT DETECTED

Important model architecture

This is not four separate violation models.

There are two trained traffic models:

25-Class YOLO Model
        +
Specialized Helmet Model

The plate detector and OCR are an additional number-plate processing stage.

📚 Dataset & Preparation

The supplied notebook shows a project dataset stored under the Google Drive project structure:

AI_PROJECT/
└── FINAL_DATASET/
    ├── train/
    ├── valid/
    ├── test/
    └── data.yaml

A balanced dataset was also prepared in the notebook:

FINAL_DATASET_BALANCED/

The notebook checked image/label counts, class distributions, sample images, and YOLO annotations before training.

The 25-class label set used in the notebook is:

0  helmet
1  no_helmet
2  motorcycle
3  triple_riding
4  phone_violation
5  right_side
6  wrong_side
7  driver
8  seatbelt
9  seatbelt_violation
10 cow
11 dog
12 buffalo
13 goat
14 chicken
15 pig
16 sheep
17 cat
18 horse
19 drinking
20 eyes_closed
21 yawning
22 nodding_off
23 looking_away
24 person

🤖 YOLO Training

The main detection work was performed with Ultralytics YOLO.

The notebook used:

YOLO11n

Image size: 640

GPU training on a Tesla T4 during the Colab training workflow

A 30-epoch training run is recorded in the notebook

Best weights were selected from the training results

The training workflow was:

Dataset
   ↓
YOLO Labels + data.yaml
   ↓
YOLO11n
   ↓
GPU Training
   ↓
Validation
   ↓
Test Evaluation
   ↓
Best Weights

The notebook also generated evaluation artifacts such as validation/test predictions and a normalized confusion matrix.

🪖 Helmet Model

A separate balanced helmet model was trained for focused helmet analysis.

helmet_balanced_best.pt

The final inference notebook uses:

HELMET_NO_HELMET_ID = 1

When class 1 is detected by the helmet model, the application records:

No Helmet

🔢 Number Plate + OCR

For images where a target violation is detected, the notebook performs an additional number-plate pipeline.

Violation
   ↓
Plate Detection
   ↓
Plate Crop
   ↓
6× Upscaling
   ↓
Grayscale / CLAHE / Sharpening
   ↓
PaddleOCR
   ↓
Text Cleaning
   ↓
Indian Plate Pattern Check
   ↓
Best OCR Candidate

The notebook uses PaddleOCR and tests multiple processed versions of the plate crop.

OCR output is cleaned to uppercase alphanumeric text and checked against Indian-style plate patterns.

Example:

Raw OCR:
KA 33 AB 1234

Cleaned:
KA33AB1234

🚨 Four Target Violations

The notebook's final inference logic uses these target classes:

Violation

Model

Class

No Helmet

Helmet Model

1

Triple Riding

25-Class YOLO

3

Phone While Driving

25-Class YOLO

4

Seatbelt Violation

25-Class YOLO

9

The other classes in the 25-class model remain part of the trained model and can be detected, but these four are the project's target violations for the final evidence/decision workflow.

🧮 Confidence-Aware Decision

The system does not treat every AI prediction as automatically correct.

A configurable confidence threshold is used by the web application:

High confidence
      ↓
DETECTED

Low confidence
      ↓
HUMAN REVIEW

No target violation
      ↓
NOT DETECTED

The deployment setting is configurable with:

REVIEW_THRESHOLD=0.60

This value can be changed without modifying the frontend.

The notebook itself used different confidence thresholds for different inference stages, including:

Main YOLO: 0.10
Helmet YOLO: 0.10
Plate detector: 0.25
OCR automatic plate check: 0.85

👨‍⚖️ Human-in-the-Loop

Human review is included because difficult traffic images can produce uncertain predictions.

Examples include:

Blurred vehicles

Partially visible riders

Poor lighting

Occluded number plates

Low OCR confidence

Ambiguous detections

The intended decision flow is:

AI Prediction
     ↓
Confidence Analysis
     ↓
 ┌───────────────┐
 │ High          │ → DETECTED
 │ Confidence    │
 └───────────────┘

 ┌───────────────┐
 │ Low           │ → HUMAN REVIEW
 │ Confidence    │
 └───────────────┘

 ┌───────────────┐
 │ No Target     │ → NOT DETECTED
 │ Violation     │
 └───────────────┘

🖥️ Web Application

The website was designed as an AI command-center rather than a basic upload form.

It provides:

Modern Indian road-safety themed visual design

Traffic image upload

Drag-and-drop image support

AI analysis button

Detection result cards

Violation count

Helmet status

Number plate result

OCR confidence

Overall confidence

Human-review status

Annotated detection image

AI engine status indicators

Responsive layout

The frontend is contained in:

index.html

The backend is:

app.py

🏗️ Backend Architecture

Browser
   │
   │ POST /api/analyze
   ▼
Flask
   │
   ├── 25-Class YOLO
   │
   ├── Helmet YOLO
   │
   ├── Plate Detector
   │
   └── PaddleOCR
   │
   ▼
Decision Engine
   │
   ├── DETECTED
   ├── HUMAN_REVIEW
   └── NOT_DETECTED
   │
   ▼
JSON Result
   │
   ▼
TrafficGuard Dashboard

🗂️ Project Structure

traffic-violation-detection-ai/
│
├── index.html
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── models/
    ├── main_25class_best.pt
    ├── helmet_balanced_best.pt
    └── plate_best.pt

Model note

The notebook confirms that the main and helmet models were restored and permanently saved during development.

The plate detector was used by the final Colab pipeline, but a plate_best.pt file was not among the two models restored in the later Drive-recovery step.

Therefore, plate detection + OCR will remain unavailable in deployment until the plate detector is added to models/plate_best.pt.

The backend intentionally does not crash if that file is missing; it reports the plate component as unavailable.

☁️ Google Drive Model Recovery Problem

Google Colab's /content directory is temporary.

During development, the application encountered errors such as:

FileNotFoundError:
25-class model NOT FOUND

and:

FileNotFoundError:
Plate model NOT FOUND

The main and helmet models were restored to Google Drive and then copied back into the runtime.

The permanent development folder became:

AI_PROJECT/
└── detection_models/
    ├── main_25class_best.pt
    └── helmet_balanced_best.pt

This solved the model-loss problem caused by restarting the Colab runtime.

🐛 Problems Faced & How We Solved Them

1. Model files disappeared after Colab restart

Problem

The trained .pt files existed in /content, but /content is temporary.

Solution

Models were copied into Google Drive and restored into the runtime when needed.

2. Incorrect model paths

Problem

The application expected files at paths such as:

/content/main_25class_best.pt
/content/helmet_balanced_best.pt
/content/plate_models/best.pt

Some files were not available at those exact locations.

Solution

The model paths were checked and the main/helmet models were placed in a common detection-model directory.

For deployment, paths are now controlled through environment variables or the local models/ directory.

3. Frontend detection request failed

Problem

The browser cannot directly execute a PyTorch/YOLO .pt model.

Solution

A Flask backend was created.

HTML / JavaScript
       ↓
Flask API
       ↓
YOLO + OCR
       ↓
JSON response
       ↓
Dashboard

4. index.html was missing from the runtime

Problem

The Colab runtime showed:

index.html: False

Solution

The frontend was made a proper repository file and is served by Flask.

5. Colab iframe / port problems

Problem

Testing the web server through the Colab iframe produced browser/security issues and an internal server error.

Solution

The application was redesigned as a normal Flask web service so it can run locally and on a deployment platform such as Render.

6. Python syntax error

Problem

An incomplete assignment caused:

SyntaxError: invalid syntax

Example of the incomplete statement:

helmet_results =

Solution

The model result was properly assigned before processing:

helmet_results = helmet_model(image)

7. Plate model unavailable

Problem

The final Colab pipeline expected:

/content/plate_models/best.pt

but that file was not restored with the two confirmed traffic models.

Solution

The web backend treats the plate detector as an optional deployment component instead of crashing the entire application.

Once plate_best.pt is placed in:

models/plate_best.pt

the plate + OCR stage becomes available.

🧪 Local Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/traffic-violation-detection-ai.git
cd traffic-violation-detection-ai

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Place the model files inside:

models/
├── main_25class_best.pt
├── helmet_balanced_best.pt
└── plate_best.pt

Run:

python app.py

Open the address shown by Flask.

🌐 Render Deployment

The application is structured as a Flask web service.

Build Command

pip install -r requirements.txt

Start Command

gunicorn app:app

Render supplies the runtime port through the PORT environment variable, which app.py reads automatically.

For deployment, make sure the required model files are actually available to the service.

Large model files should be handled carefully because hosting platforms can impose repository, build, storage, memory, and disk constraints.

🔧 Environment Variables

The backend supports:

MODEL_DIR
MAIN_MODEL_PATH
HELMET_MODEL_PATH
PLATE_MODEL_PATH
MAIN_CONF
HELMET_CONF
PLATE_CONF
REVIEW_THRESHOLD
CAMERA_LOCATION

Example:

REVIEW_THRESHOLD=0.60
CAMERA_LOCATION=Camera 01

⚠️ Current Limitations

TrafficGuard AI is an AI-assisted prototype.

Performance can be affected by:

Dataset quality

Image resolution

Lighting

Motion blur

Camera angle

Occlusion

Object size

Number-plate visibility

OCR quality

Model confidence

A confidence score is not a guarantee of correctness.

For real-world traffic enforcement, additional validation, privacy protection, security, auditability, and human oversight would be required.

🔮 Future Improvements

Possible next stages include:

Real-time CCTV/video analysis

Vehicle tracking

Improved Indian number-plate OCR

Advanced reviewer dashboard

Reviewer corrections

Analytics and violation trends

Authentication and role-based access

Secure evidence workflows

Better low-light detection

Model optimization for cloud deployment

GPU/CPU deployment optimization

📌 Project Status

Component

Status

Traffic dataset preparation

✅

YOLO 25-class training

✅

Main model

✅

Helmet model

✅

Model recovery through Drive

✅

Flask backend

✅

Modern dashboard

✅

Four target violations

✅

Number plate detection

🔄 Requires plate model in deployment

PaddleOCR

🔄 Deployment dependent

Human review decision path

✅

Render deployment

🔄

👨‍💻 Author

Karthik Kumar

TrafficGuard AI

Intelligent AI-assisted road safety analysis.

🇮🇳 Vision

TrafficGuard AI demonstrates how computer vision can be combined with specialized AI models and human verification to create a more responsible traffic-safety workflow.

        DETECT
           ↓
        ANALYZE
           ↓
      READ PLATE
           ↓
       CHECK OCR
           ↓
   MEASURE CONFIDENCE
           ↓
   ┌───────┼────────┐
   ↓       ↓        ↓
DETECTED REVIEW  NOT DETECTED
           ↓
     HUMAN DECISION

🚦 TrafficGuard AI

Detect. Analyze. Verify. Improve Road Safety.

Designed & Developed by Karthik Kumar 🇮🇳
