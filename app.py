import os
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ============================================================
# TRAFFICGUARD AI
# Flask + YOLO Backend
# ============================================================

app = Flask(__name__)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "results"

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
    "mp4",
    "avi",
    "mov",
    "mkv",
}

# ------------------------------------------------------------
# Model paths
# ------------------------------------------------------------

MAIN_MODEL_PATH = BASE_DIR / "main_25class_best.pt"
HELMET_MODEL_PATH = BASE_DIR / "helmet_balanced_best.pt"

print("=" * 70)
print("TRAFFICGUARD AI - MODEL LOADING")
print("=" * 70)

print("Main model:")
print(MAIN_MODEL_PATH)
print("Exists:", MAIN_MODEL_PATH.exists())

print("\nHelmet model:")
print(HELMET_MODEL_PATH)
print("Exists:", HELMET_MODEL_PATH.exists())

# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------

if not MAIN_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Main model not found: {MAIN_MODEL_PATH}"
    )

if not HELMET_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Helmet model not found: {HELMET_MODEL_PATH}"
    )

print("\nLoading 25-class model...")
main_model = YOLO(str(MAIN_MODEL_PATH))
print("✅ 25-class model loaded")

print("\nLoading helmet model...")
helmet_model = YOLO(str(HELMET_MODEL_PATH))
print("✅ Helmet model loaded")

print("\n🎉 BOTH MODELS ARE READY")
print("=" * 70)


# ============================================================
# Helper functions
# ============================================================

def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    if not filename:
        return False

    extension = filename.rsplit(".", 1)[-1].lower()

    return extension in ALLOWED_EXTENSIONS


def convert_detections(result):
    """
    Convert Ultralytics detection result into JSON-safe data.
    """

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        # Get class name safely
        class_name = result.names.get(
            class_id,
            str(class_id)
        )

        coordinates = box.xyxy[0].tolist()

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "confidence_percent": round(
                confidence * 100,
                2
            ),
            "bbox": {
                "x1": round(coordinates[0], 2),
                "y1": round(coordinates[1], 2),
                "x2": round(coordinates[2], 2),
                "y2": round(coordinates[3], 2),
            }
        })

    return detections


def save_annotated_result(result, output_path):
    """Save YOLO annotated image/video result."""

    plotted = result.plot()

    import cv2

    cv2.imwrite(
        str(output_path),
        plotted
    )


# ============================================================
# Routes
# ============================================================

@app.route("/")
def home():
    """Serve the TrafficGuard AI frontend."""

    return send_from_directory(
        str(BASE_DIR),
        "index.html"
    )


@app.route("/health")
def health():
    """Health check endpoint for Render."""

    return jsonify({
        "status": "online",
        "service": "TrafficGuard AI",
        "main_model": MAIN_MODEL_PATH.exists(),
        "helmet_model": HELMET_MODEL_PATH.exists()
    })


# ============================================================
# Image Analysis API
# ============================================================

@app.route("/api/analyze", methods=["POST"])
def analyze():

    try:

        # ----------------------------------------------------
        # Check upload
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No file uploaded."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        if not allowed_file(file.filename):

            return jsonify({
                "success": False,
                "error": "Unsupported file type."
            }), 400

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        original_name = secure_filename(
            file.filename
        )

        unique_name = (
            uuid.uuid4().hex
            + "_"
            + original_name
        )

        input_path = (
            UPLOAD_FOLDER
            / unique_name
        )

        file.save(str(input_path))

        print("\n" + "=" * 70)
        print("NEW ANALYSIS")
        print("=" * 70)
        print("File:", original_name)

        # ----------------------------------------------------
        # Run 25-class model
        # ----------------------------------------------------

        print("Running 25-class model...")

        main_results = main_model.predict(
            source=str(input_path),
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        # ----------------------------------------------------
        # Run helmet model
        # ----------------------------------------------------

        print("Running helmet model...")

        helmet_results = helmet_model.predict(
            source=str(input_path),
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        # ----------------------------------------------------
        # Convert results
        # ----------------------------------------------------

        main_detections = []

        for result in main_results:
            main_detections.extend(
                convert_detections(result)
            )

        helmet_detections = []

        for result in helmet_results:
            helmet_detections.extend(
                convert_detections(result)
            )

        # ----------------------------------------------------
        # Save annotated output
        # ----------------------------------------------------

        result_filename = (
            Path(unique_name).stem
            + "_result.jpg"
        )

        result_path = (
            RESULT_FOLDER
            / result_filename
        )

        if main_results:

            save_annotated_result(
                main_results[0],
                result_path
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_main = len(main_detections)
        total_helmet = len(helmet_detections)

        # Count classes
        class_counts = {}

        for detection in main_detections:

            name = detection["class_name"]

            class_counts[name] = (
                class_counts.get(name, 0) + 1
            )

        # Helmet counts
        helmet_class_counts = {}

        for detection in helmet_detections:

            name = detection["class_name"]

            helmet_class_counts[name] = (
                helmet_class_counts.get(name, 0) + 1
            )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response = {
            "success": True,

            "filename": original_name,

            "result_url": (
                "/results/"
                + result_filename
            ),

            "summary": {
                "total_detections": total_main,
                "helmet_detections": total_helmet,
            },

            "class_counts": class_counts,

            "helmet_class_counts":
                helmet_class_counts,

            "detections": main_detections,

            "helmet_detections":
                helmet_detections,
        }

        print("\nAnalysis completed.")
        print("Main detections:", total_main)
        print("Helmet detections:", total_helmet)

        return jsonify(response)

    except Exception as error:

        print("\n❌ ANALYSIS ERROR")
        print(str(error))

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ============================================================
# Serve result files
# ============================================================

@app.route("/results/<filename>")
def serve_result(filename):

    return send_from_directory(
        str(RESULT_FOLDER),
        filename
    )


# ============================================================
# Optional upload route
# ============================================================

@app.route("/api/status")
def status():

    return jsonify({
        "system": "TrafficGuard AI",
        "status": "online",
        "models": {
            "25_class": MAIN_MODEL_PATH.exists(),
            "helmet": HELMET_MODEL_PATH.exists()
        }
    })


# ============================================================
# Run Flask
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    print("\n" + "=" * 70)
    print("TRAFFICGUARD AI SERVER")
    print("=" * 70)
    print(f"Starting server on port {port}")
    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
