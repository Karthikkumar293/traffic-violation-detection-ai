import os
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Limit uploaded images to 15 MB
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("TrafficGuard")


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL FILES
# ============================================================

MAIN_MODEL_NAME = "main_25class_best.pt"
HELMET_MODEL_NAME = "helmet_balanced_best.pt"


def find_model(filename):
    """
    Look for the model in:
        1. Repository root
        2. models/ directory
    """

    possible_paths = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
    ]

    for path in possible_paths:

        if path.is_file():

            logger.info("Model found: %s", path)

            return path

    logger.error("Model not found: %s", filename)

    return None


# ============================================================
# LOAD MODELS
# ============================================================

logger.info("=" * 60)
logger.info("TRAFFICGUARD AI MODEL LOADING")
logger.info("=" * 60)


MAIN_MODEL_PATH = find_model(MAIN_MODEL_NAME)
HELMET_MODEL_PATH = find_model(HELMET_MODEL_NAME)


main_model = None
helmet_model = None


# ------------------------------------------------------------
# MAIN MODEL
# ------------------------------------------------------------

if MAIN_MODEL_PATH:

    try:

        logger.info("Loading main 25-class model...")

        main_model = YOLO(str(MAIN_MODEL_PATH))

        logger.info("Main model loaded successfully")

    except Exception as e:

        logger.exception(
            "Failed to load main model: %s",
            e
        )

else:

    logger.error("Main model is unavailable")


# ------------------------------------------------------------
# HELMET MODEL
# ------------------------------------------------------------

if HELMET_MODEL_PATH:

    try:

        logger.info("Loading helmet model...")

        helmet_model = YOLO(str(HELMET_MODEL_PATH))

        logger.info("Helmet model loaded successfully")

    except Exception as e:

        logger.exception(
            "Failed to load helmet model: %s",
            e
        )

else:

    logger.error("Helmet model is unavailable")


logger.info("=" * 60)
logger.info("MODEL RESTORATION")
logger.info("=" * 60)
logger.info("Main model : %s", main_model is not None)
logger.info("Helmet model: %s", helmet_model is not None)
logger.info("=" * 60)


# ============================================================
# HELPERS
# ============================================================

def get_class_name(model, class_id):
    """
    Safely get YOLO class name.
    """

    try:

        names = model.names

        if isinstance(names, dict):
            return str(names.get(class_id, f"class_{class_id}"))

        if isinstance(names, list):
            if 0 <= class_id < len(names):
                return str(names[class_id])

    except Exception:
        pass

    return f"class_{class_id}"


def run_model(model, image, confidence=0.25):
    """
    Run YOLO safely and return detection information.
    """

    detections = []

    if model is None:
        return detections

    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=416,
        verbose=False
    )

    if not results:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    boxes = result.boxes

    for i in range(len(boxes)):

        try:

            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            xyxy = boxes.xyxy[i].cpu().numpy().tolist()

            x1, y1, x2, y2 = [
                int(round(x))
                for x in xyxy
            ]

            class_name = get_class_name(
                model,
                cls_id
            )

            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            })

        except Exception as e:

            logger.warning(
                "Could not parse detection: %s",
                e
            )

    return detections


def draw_detections(
    image,
    detections,
    label_prefix=""
):
    """
    Draw bounding boxes on image.
    """

    output = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = (
            f"{label_prefix}{class_name} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        text_y = max(y1 - 8, th + 5)

        cv2.rectangle(
            output,
            (x1, text_y - th - baseline),
            (x1 + tw + 6, text_y + 3),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            output,
            label,
            (x1 + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return output


def image_to_data_url(image):
    """
    Convert OpenCV image to base64 data URL.
    This allows the frontend to display the result
    without requiring permanent file storage.
    """

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )

    if not success:
        return None

    encoded_bytes = encoded.tobytes()

    encoded_base64 = base64.b64encode(
        encoded_bytes
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded_base64
    )


def determine_status(
    main_detections,
    helmet_detections
):
    """
    Basic result classification.

    The model's actual class names are preserved.
    We do not invent custom violation classes.
    """

    violations = []

    # Helmet model classes
    for detection in helmet_detections:

        name = detection["class_name"].lower()

        # Common helmet dataset naming
        if (
            "no_helmet" in name
            or "without_helmet" in name
            or "no helmet" in name
            or name in [
                "withouthelmet",
                "nohelmet"
            ]
        ):

            violations.append({
                "type": "No Helmet",
                "confidence": detection["confidence"]
            })

    if violations:

        return "Violation Detected", violations

    if main_detections or helmet_detections:

        return "Detection Complete", violations

    return "No Detection", violations


# ============================================================
# ROOT ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    index_file = BASE_DIR / "index.html"

    if index_file.exists():

        return send_from_directory(
            BASE_DIR,
            "index.html"
        )

    return """
    <html>
        <head>
            <title>TrafficGuard AI</title>
        </head>
        <body>
            <h1>TrafficGuard AI</h1>
            <p>Service is online.</p>
            <p>index.html was not found.</p>
        </body>
    </html>
    """


# ============================================================
# HEALTH API
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "service": "TrafficGuard AI",
        "status": "online",
        "main_model": main_model is not None,
        "helmet_model": helmet_model is not None,
        "ocr": False,
        "plate_model": False,
        "inference_size": 416,
        "startup_warnings": []
    })


# ============================================================
# ANALYZE API
# ============================================================

@app.route("/api/analyze", methods=["POST"])
def analyze():

    try:

        # ----------------------------------------------------
        # CHECK MAIN MODEL
        # ----------------------------------------------------

        if main_model is None:

            return jsonify({
                "success": False,
                "status": "Analysis failed",
                "error": "Main YOLO model is not loaded."
            }), 503


        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "status": "Analysis failed",
                "error": "No image file was uploaded."
            }), 400


        uploaded_file = request.files["file"]


        if not uploaded_file:

            return jsonify({
                "success": False,
                "status": "Analysis failed",
                "error": "Invalid uploaded file."
            }), 400


        if uploaded_file.filename == "":

            return jsonify({
                "success": False,
                "status": "Analysis failed",
                "error": "No filename provided."
            }), 400


        logger.info(
            "Analysis request received: %s",
            uploaded_file.filename
        )


        # ----------------------------------------------------
        # READ IMAGE DIRECTLY
        # ----------------------------------------------------

        file_bytes = uploaded_file.read()

        if not file_bytes:

            return jsonify({
                "success": False,
                "status": "Analysis failed",
                "error": "Uploaded file is empty."
            }), 400


        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return jsonify({
                "success": False,
                "status": "Analysis failed",
                "error": "Uploaded file is not a valid image."
            }), 400


        height, width = image.shape[:2]


        logger.info(
            "Image received: %sx%s",
            width,
            height
        )


        # ----------------------------------------------------
        # MAIN MODEL DETECTION
        # ----------------------------------------------------

        logger.info(
            "Running main model..."
        )

        main_detections = run_model(
            main_model,
            image,
            confidence=0.25
        )


        logger.info(
            "Main detections: %s",
            len(main_detections)
        )


        # ----------------------------------------------------
        # HELMET MODEL DETECTION
        # ----------------------------------------------------

        helmet_detections = []

        if helmet_model is not None:

            logger.info(
                "Running helmet model..."
            )

            helmet_detections = run_model(
                helmet_model,
                image,
                confidence=0.25
            )

            logger.info(
                "Helmet detections: %s",
                len(helmet_detections)
            )


        # ----------------------------------------------------
        # DRAW RESULTS
        # ----------------------------------------------------

        annotated = draw_detections(
            image,
            main_detections,
            label_prefix=""
        )

        annotated = draw_detections(
            annotated,
            helmet_detections,
            label_prefix="Helmet: "
        )


        # ----------------------------------------------------
        # RESULT STATUS
        # ----------------------------------------------------

        status, violations = determine_status(
            main_detections,
            helmet_detections
        )


        # ----------------------------------------------------
        # ALL DETECTIONS
        # ----------------------------------------------------

        all_detections = (
            main_detections
            + helmet_detections
        )


        confidences = [
            d["confidence"]
            for d in all_detections
        ]


        if confidences:

            overall_confidence = (
                sum(confidences)
                / len(confidences)
            )

        else:

            overall_confidence = 0.0


        # ----------------------------------------------------
        # NUMBER PLATE
        # ----------------------------------------------------

        plate_detection = None

        for detection in main_detections:

            name = detection["class_name"].lower()

            if (
                "plate" in name
                or "license" in name
                or "number" in name
            ):

                plate_detection = detection
                break


        if plate_detection:

            number_plate = plate_detection["class_name"]
            plate_confidence = plate_detection["confidence"]

        else:

            number_plate = "Not detected"
            plate_confidence = None


        # ----------------------------------------------------
        # ENCODE IMAGE
        # ----------------------------------------------------

        result_image = image_to_data_url(
            annotated
        )


        if result_image is None:

            return jsonify({
                "success": False,
                "status": "Analysis failed",
                "error": "Could not encode result image."
            }), 500


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "success": True,

            "status": status,

            "analysis_status": status,

            "overall_confidence": round(
                overall_confidence,
                4
            ),

            "violation_type": (
                violations[0]["type"]
                if violations
                else "-"
            ),

            "number_plate": number_plate,

            "ocr_confidence": None,

            "plate_confidence": plate_confidence,

            "detected_violations": len(
                violations
            ),

            "violations": violations,

            "detections": all_detections,

            "main_detections": main_detections,

            "helmet_detections": helmet_detections,

            "image_width": width,

            "image_height": height,

            "annotated_image": result_image,

            "result_image": result_image
        }


        logger.info(
            "Analysis completed successfully"
        )

        return jsonify(response)


    except Exception as e:

        logger.exception(
            "Analysis failed"
        )

        return jsonify({

            "success": False,

            "status": "Analysis failed",

            "error": str(e)

        }), 500


# ============================================================
# FILE SIZE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "status": "Analysis failed",

        "error": "Image is too large. Maximum size is 15 MB."

    }), 413


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.errorhandler(Exception)
def handle_exception(error):

    logger.exception(
        "Unhandled server error"
    )

    return jsonify({

        "success": False,

        "status": "Server error",

        "error": str(error)

    }), 500


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
