# 🚦 TrafficGuard AI

## 🇮🇳 Intelligent Traffic Violation Detection & Road Safety Platform
## LIVE DEMO https://traffic-violation-detection-ai.onrender.com

TrafficGuard AI is an AI-powered computer vision system designed to automatically analyze traffic images and identify road-safety violations.

The system combines a **25-class YOLO detection model**, a **specialized helmet detection model**, **number-plate detection**, **OCR**, confidence-based decision making, and **human review for uncertain detections**.

The goal is to build an AI-assisted traffic monitoring system that can reduce the effort required for manual traffic-image analysis while keeping human verification available when the AI prediction is uncertain.

---

## 🎯 Project Objective

Traditional traffic monitoring can require significant manual effort when a large number of road images need to be inspected.

TrafficGuard AI attempts to automate the initial analysis by using computer vision models to identify important traffic violations and extract vehicle number-plate information.

The system is designed to:

- Detect traffic violations automatically.
- Analyze multiple traffic-related classes using a 25-class YOLO model.
- Perform specialized helmet detection using a dedicated helmet model.
- Detect vehicle number plates.
- Extract number-plate text using OCR.
- Calculate detection confidence.
- Identify low-confidence predictions.
- Send uncertain detections for human review.
- Clearly identify images where no target violation is detected.
- Display the complete result through a modern web dashboard.

---

# 🧠 Complete System Workflow

The complete TrafficGuard AI pipeline is:

    ┌───────────────────────────────┐
    │       TRAFFIC IMAGE           │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │     25-CLASS YOLO MODEL        │
    │     main_25class_best.pt       │
    └───────────────┬───────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │ Four Target Violations  │
        │                         │
        │ • Triple Riding         │
        │ • Phone While Driving   │
        │ • Seatbelt Violation    │
        │ • No Helmet             │
        └────────────┬────────────┘
                     │
                     │
             ┌───────▼────────┐
             │ Helmet Model   │
             │ Specialized    │
             │ Analysis       │
             └───────┬────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Number Plate        │
          │ Detection           │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ OCR                 │
          │ Plate Text          │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Confidence Analysis │
          └──────────┬──────────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
      DETECTED   HUMAN REVIEW   NOT DETECTED

---

# 🤖 AI Models

TrafficGuard AI uses **two trained YOLO models**.

There are not four separate models for the four violations.

The primary system uses one 25-class model, while a second specialized model is used for helmet analysis.

---

## 1. 25-Class YOLO Model

The main detection model is:

    main_25class_best.pt

This is the primary YOLO model trained to recognize the traffic-related classes used by the project.

The model contains **25 classes**.

The four main violation categories used by the application are:

| Violation | Model |
|---|---|
| Triple Riding | 25-Class YOLO |
| Phone While Driving | 25-Class YOLO |
| Seatbelt Violation | 25-Class YOLO |
| No Helmet | Specialized Helmet Model |

The main model is therefore the central detection model of the system.

---

## 2. Helmet Detection Model

A separate specialized model is used for helmet analysis:

    helmet_balanced_best.pt

The purpose of this model is to provide a dedicated helmet-related detection stage.

The two-model approach allows the project to use the 25-class traffic model for general traffic analysis while using a specialized model for helmet detection.

---

# 📊 Dataset

The project uses traffic-related image data for training and evaluating the computer vision models.

The dataset preparation process included:

- Collecting traffic images.
- Organizing images and annotations.
- Preparing YOLO-compatible labels.
- Checking annotations.
- Organizing classes.
- Preparing training and validation data.
- Removing or correcting problematic samples.
- Training the YOLO models.
- Testing the trained models on traffic images.

Dataset quality was one of the most important factors during development because object-detection performance depends strongly on the quality and diversity of the training data.

---

# 🏋️ Model Training

The general training pipeline was:

    Traffic Dataset
          ↓
    Image & Label Preparation
          ↓
    YOLO Dataset Configuration
          ↓
    Model Training
          ↓
    Validation
          ↓
    Best Weight Selection
          ↓
    Inference Testing

The trained model weights were saved as:

    main_25class_best.pt

and:

    helmet_balanced_best.pt

The best-performing weights were then restored from persistent storage when the Google Colab runtime was restarted.

---

# 🔢 Number Plate Detection + OCR

Number-plate recognition is implemented as a separate stage after the traffic/violation analysis.

The workflow is:

    Traffic Image
         ↓
    Number Plate Detection
         ↓
    Plate Bounding Box
         ↓
    Plate Crop
         ↓
    Image Preprocessing
         ↓
    OCR
         ↓
    Number Plate Text
         ↓
    OCR Confidence

For example:

    Detected Plate:
    KA01AB1234

The OCR result is accompanied by a confidence value where available.

OCR performance can be affected by:

- Low image resolution.
- Motion blur.
- Poor lighting.
- Vehicle distance.
- Plate angle.
- Occlusion.
- Dirty or damaged plates.

Therefore, OCR results should be considered AI-generated information that may require verification.

---

# 📈 Confidence-Based Decision System

TrafficGuard AI does not blindly treat every model prediction as correct.

The system considers the confidence of the detected violation.

The basic decision flow is:

    AI Detection
         ↓
    Confidence Analysis
         ↓
    ┌──────────────┬──────────────────┐
    │              │                  │
    ▼              ▼                  ▼
    HIGH          LOW             NO TARGET
 CONFIDENCE    CONFIDENCE         DETECTION
    │              │                  │
    ▼              ▼                  ▼
 DETECTED     HUMAN REVIEW       NOT DETECTED

This provides a safer workflow than automatically treating every prediction as a confirmed violation.

---

# 👨‍⚖️ Human Review

One of the main design features of TrafficGuard AI is the **Human-in-the-Loop** concept.

If the model produces an uncertain prediction, the system can classify the case as:

    HUMAN REVIEW

The reviewer can then inspect the image and decide whether the AI prediction is correct.

This is particularly important for difficult images containing:

- Occluded vehicles.
- Small objects.
- Blurred vehicles.
- Poor lighting.
- Crowded traffic.
- Partially visible number plates.

The objective is:

    AI ASSISTS
         ↓
    HUMAN VERIFIES WHEN NECESSARY

rather than assuming that AI predictions are always correct.

---

# ❌ Not Detected

TrafficGuard AI also handles images where none of the target violations are detected.

The workflow is:

    Traffic Image
         ↓
    AI Analysis
         ↓
    No Target Violation
         ↓
    NOT DETECTED

This is important because the system should not classify every uploaded traffic image as a violation.

---

# ⚠️ Problems Faced During Development

Several technical problems were encountered while developing the project.

These problems helped shape the final architecture.

---

## 1. Dataset Preparation Issues

Initially, dataset preparation required considerable work to ensure that images and annotations were suitable for YOLO training.

Problems included:

- Incorrect annotations.
- Class organization.
- Different image resolutions.
- Difficult traffic scenes.
- Poor-quality images.
- Inconsistent labels.

### Solution

The dataset was reorganized and the annotations were prepared according to the YOLO dataset structure before training and validation.

---

## 2. Model Files Were Lost After Colab Restart

Google Colab provides a temporary runtime environment.

Initially, model files were stored inside:

    /content/

For example:

    /content/main_25class_best.pt

When the runtime restarted, the files were no longer guaranteed to exist.

This caused errors such as:

    FileNotFoundError:
    25-class model NOT FOUND

### Solution

Google Drive was connected and the trained models were permanently stored there.

The models were restored into the Colab environment whenever necessary.

The permanent model directory became:

    AI_PROJECT/
    └── detection_models/
        ├── main_25class_best.pt
        └── helmet_balanced_best.pt

This solved the model-loss problem caused by Colab runtime resets.

---

## 3. Helmet Model Path Problem

The application initially expected:

    /content/helmet_balanced_best.pt

After a runtime restart, the model was not available at that location.

### Solution

The model was restored from Google Drive and copied back into the active environment.

The restored model was then verified before loading it into YOLO.

---

## 4. Plate Model Path Problem

The application expected the plate model at:

    /content/plate_models/best.pt

At one point, this file was not available at the expected location.

This produced:

    FileNotFoundError:
    Plate model NOT FOUND

### Solution

The plate model was separated from the main model recovery process.

The application was designed so that the main 25-class model and helmet model could be restored and verified independently while the plate-detection component was integrated separately.

---

## 5. Frontend Cannot Directly Run YOLO `.pt` Models

A major architectural problem was that the HTML/JavaScript frontend cannot directly execute a PyTorch YOLO `.pt` model in a normal browser.

### Solution

A Python Flask backend was introduced.

The final communication architecture became:

    Browser
       ↓
    JavaScript
       ↓
    Flask API
       ↓
    YOLO Models
       ↓
    OCR
       ↓
    Detection Result
       ↓
    Flask JSON Response
       ↓
    Web Dashboard

This separates the user interface from the AI inference system.

---

## 6. `index.html` Was Not Available

During Colab development, the system showed:

    index.html: False

while the model files were available.

### Solution

The frontend was placed directly inside the project repository as:

    index.html

The frontend can then communicate with the Flask backend.

---

## 7. Colab Port / Browser Issue

The project was initially tested through Google Colab.

The application was exposed using a Colab port.

However, browser and security restrictions caused errors such as:

    Internal Server Error

### Solution

The project architecture was moved toward a standard Flask web application that can run as an independent service.

This makes it more suitable for deployment using a platform such as Render.

---

## 8. Python Syntax Error

During development, an incomplete assignment caused a Python syntax error:

    helmet_results =

### Solution

The model inference code was corrected so that model results were properly assigned before processing.

---

# 🏗️ System Architecture

The complete application architecture is:

    ┌───────────────────────────┐
    │          USER             │
    └─────────────┬─────────────┘
                  │
                  ▼
    ┌───────────────────────────┐
    │       WEB FRONTEND        │
    │                           │
    │ HTML + CSS + JavaScript   │
    │        index.html         │
    └─────────────┬─────────────┘
                  │
                  │ POST /api/analyze
                  ▼
    ┌───────────────────────────┐
    │       FLASK BACKEND       │
    │          app.py           │
    └─────────────┬─────────────┘
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
    ┌──────────────┐  ┌──────────────┐
    │ 25-Class     │  │ Helmet       │
    │ YOLO Model   │  │ YOLO Model   │
    └──────┬───────┘  └──────┬───────┘
           │                  │
           └────────┬─────────┘
                    ▼
          ┌────────────────────┐
          │ Violation Analysis │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Plate Detection    │
          │ + OCR              │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Confidence Engine  │
          └──────────┬─────────┘
                     │
           ┌─────────┼─────────┐
           ▼         ▼         ▼
       DETECTED    REVIEW   NOT DETECTED

---

# 🖥️ Web Application

TrafficGuard AI includes a modern traffic-command-center style web dashboard.

The website is designed to provide a professional interface rather than a basic machine-learning demonstration.

The interface includes:

- Image upload.
- Drag-and-drop support.
- AI analysis.
- Detection visualization.
- Violation information.
- Confidence information.
- Helmet status.
- Number-plate information.
- OCR output.
- Human-review status.
- Not-detected status.
- System status indicators.
- Annotated image display.

---

# 🎨 Dashboard

The main dashboard is organized into:

    TRAFFICGUARD AI

    ├── Overview
    ├── Detection
    ├── Evidence
    ├── Human Review
    └── Analytics

The analysis area provides information such as:

    Violations
    Helmet Status
    Number Plate
    Confidence
    OCR Result
    Review Status

The UI uses an Indian traffic/road-safety visual identity to represent the project's intended environment.

---

# 📁 Project Structure

The project structure is:

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

The exact model-file handling may be changed during deployment because large `.pt` files may not be suitable for a normal GitHub repository.

---

# 🧰 Technologies Used

## Programming

- Python
- HTML5
- CSS3
- JavaScript

## Machine Learning

- YOLO
- PyTorch
- Computer Vision
- Object Detection
- Confidence Analysis

## AI Models

- 25-Class Traffic Detection Model
- Specialized Helmet Detection Model

## Number Plate Recognition

- Number Plate Detection
- OCR
- Image Processing

## Backend

- Flask
- REST API

## Development

- Google Colab
- Google Drive
- Git
- GitHub

## Deployment

- Render
- Gunicorn

---

# 🚀 Deployment

The project is designed to be deployed as a Flask web application.

The production architecture is:

    User
      ↓
    Render
      ↓
    Flask
      ↓
    AI Models
      ↓
    OCR
      ↓
    Detection Result
      ↓
    Browser

The application can be started using Gunicorn:

    gunicorn app:app

Dependencies are installed using:

    pip install -r requirements.txt

The Flask application uses the deployment platform's `PORT` environment variable.

---

# 🔐 Privacy & Responsible AI

Traffic images and number plates may contain potentially sensitive information.

For a real-world production deployment, appropriate security and privacy controls should be implemented.

These may include:

- HTTPS.
- Authentication.
- Authorization.
- Secure image processing.
- Access control.
- Data-retention policies.
- Audit logging.
- Human verification.
- Protection of sensitive information.

AI-generated predictions should not automatically be treated as legally conclusive evidence without appropriate validation and human oversight.

---

# ⚠️ Limitations

The performance of TrafficGuard AI depends on the quality of the input image and training data.

Potential limitations include:

- Poor lighting.
- Motion blur.
- Low-resolution images.
- Small objects.
- Occlusion.
- Crowded traffic.
- Unusual camera angles.
- Partially visible vehicles.
- Difficult number plates.
- OCR errors.

A high confidence score does not guarantee that a prediction is correct.

A low confidence score does not necessarily mean that the prediction is incorrect.

Therefore, human verification remains important for uncertain cases.

---

# 🔮 Future Improvements

Future versions of TrafficGuard AI can include:

## 🎥 Real-Time CCTV Detection

Integration with live CCTV and traffic-camera streams.

## 🚘 Vehicle Tracking

Tracking vehicles across multiple video frames.

## 📊 Advanced Analytics

Analytics for:

- Violation frequency.
- Helmet compliance.
- Vehicle categories.
- Confidence distribution.
- Time-based traffic trends.
- Location-based analysis.

## 👨‍⚖️ Improved Human Review

Reviewers could:

- Confirm violations.
- Reject false detections.
- Correct OCR results.
- Add review comments.
- Finalize uncertain cases.

## 🔢 Improved Indian Number Plate OCR

Future OCR improvements can target:

- Low-light plates.
- Motion-blurred plates.
- Angled plates.
- Partially blocked plates.
- Different Indian registration formats.

## 🧠 Model Improvements

Training on larger and more diverse traffic datasets can improve generalization to real-world traffic scenes.

---

# 📌 Development Status

| Component | Status |
|---|---|
| Traffic Dataset Preparation | ✅ Completed |
| 25-Class YOLO Model | ✅ Available |
| Helmet Detection Model | ✅ Available |
| Google Drive Model Recovery | ✅ Completed |
| Model Verification | ✅ Completed |
| Modern Web Dashboard | ✅ Completed |
| Image Upload | ✅ Completed |
| AI Analysis API | ✅ Implemented |
| Four Violation Analysis | ✅ Implemented |
| Number Plate Detection | 🔄 Integration / Testing |
| OCR | 🔄 Integration / Testing |
| Human Review Workflow | 🔄 Integration / Testing |
| Render Deployment | 🔄 Preparation |
| Advanced Analytics | 🔮 Planned |

---

# 🏆 Why TrafficGuard AI?

TrafficGuard AI is designed as more than a simple:

    Image → Model → Prediction

application.

It combines:

    25-Class YOLO
          +
    Helmet Detection
          +
    Number Plate Detection
          +
    OCR
          +
    Confidence Analysis
          +
    Human Review

into one traffic-safety workflow.

The system is designed around the principle:

    DETECT
       ↓
    ANALYZE
       ↓
    MEASURE CONFIDENCE
       ↓
    ┌───────────────┬────────────────┐
    │               │                │
    ▼               ▼                ▼
 CERTAIN         UNCERTAIN        NO VIOLATION
    │               │                │
    ▼               ▼                ▼
 RESULT        HUMAN REVIEW     NOT DETECTED

This approach allows AI to perform the initial analysis while keeping human verification available for uncertain cases.

---

# 💡 Core Philosophy

The central idea of TrafficGuard AI is:

> **AI should assist traffic-safety analysis, not blindly replace human judgment.**

The system therefore combines automated computer vision with confidence-based decision making and human review.

---

# 👨‍💻 Author

## Karthik Kumar

**TrafficGuard AI**

AI-Powered Traffic Violation Detection & Road Safety Platform

---

# 🇮🇳 Vision

TrafficGuard AI aims to demonstrate how artificial intelligence and computer vision can be applied to intelligent traffic monitoring and road safety.

By combining:

    YOLO
      +
    Computer Vision
      +
    Helmet Detection
      +
    Number Plate Detection
      +
    OCR
      +
    Confidence Analysis
      +
    Human Review

TrafficGuard AI provides a complete AI-assisted traffic violation analysis workflow.

---

## 🚦 TRAFFICGUARD AI

### Detect. Analyze. Verify. Improve Road Safety.

**Built with AI, Computer Vision and Human Intelligence. 🇮🇳**
