import os
import base64
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO


# ============================================================
# TRAFFICGUARD AI
# Flask + YOLO backend
# ============================================================

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrafficGuard")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.getenv(
    "MODEL_DIR",
    os.path.join(BASE_DIR, "models")
)

MAIN_MODEL_PATH = os.getenv(
    "MAIN_MODEL_PATH",
    os.path.join(MODEL_DIR, "main_25class_best.pt")
)

HELMET_MODEL_PATH = os.getenv(
    "HELMET_MODEL_PATH",
    os.path.join(MODEL_DIR, "helmet_balanced_best.pt")
)

MAX_IMAGE_SIZE = 12 * 1024 * 1024

INFERENCE_SIZE = int(
    os.getenv("INFERENCE_SIZE", "416")
)

MAIN_CONF = float(
    os.getenv("MAIN_CONF", "0.25")
)

HELMET_CONF = float(
    os.getenv("HELMET_CONF", "0.25")
)

REVIEW_THRESHOLD = float(
    os.getenv("REVIEW_THRESHOLD", "0.60")
)


# ============================================================
# TARGET VIOLATIONS
# ============================================================

TARGET_CLASSES = {
    3: "Triple Riding",
    4: "Phone While Driving",
    9: "Seatbelt Violation",
}

HELMET_NO_HELMET_ID = 1


# ============================================================
# MODEL VARIABLES
# ============================================================

main_model = None
helmet_model = None

main_model_error = None
helmet_model_error = None


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    global main_model
    global helmet_model
    global main_model_error
    global helmet_model_error

    logger.info("=" * 60)
    logger.info("TRAFFICGUARD AI MODEL LOADING")
    logger.info("=" * 60)

    # -----------------------------
    # Main model
    # -----------------------------

    if os.path.isfile(MAIN_MODEL_PATH):

        try:

            logger.info(
                "Loading main model: %s",
                MAIN_MODEL_PATH
            )

            main_model = YOLO(MAIN_MODEL_PATH)

            logger.info(
                "Main model loaded successfully"
            )

        except Exception as e:

            main_model_error = str(e)

            logger.exception(
                "Main model failed to load"
            )

    else:

        main_model_error = (
            "Main model not found: "
            + MAIN_MODEL_PATH
        )

        logger.error(main_model_error)


    # -----------------------------
    # Helmet model
    # -----------------------------

    if os.path.isfile(HELMET_MODEL_PATH):

        try:

            logger.info(
                "Loading helmet model: %s",
                HELMET_MODEL_PATH
            )

            helmet_model = YOLO(
                HELMET_MODEL_PATH
            )

            logger.info(
                "Helmet model loaded successfully"
            )

        except Exception as e:

            helmet_model_error = str(e)

            logger.exception(
                "Helmet model failed to load"
            )

    else:

        helmet_model_error = (
            "Helmet model not found: "
            + HELMET_MODEL_PATH
        )

        logger.error(helmet_model_error)


    logger.info("=" * 60)
    logger.info(
        "Main model: %s",
        main_model is not None
    )

    logger.info(
        "Helmet model: %s",
        helmet_model is not None
    )

    logger.info("=" * 60)


load_models()


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "online",

        "service": "TrafficGuard AI",

        "main_model": main_model is not None,

        "helmet_model": helmet_model is not None,

        "plate_model": False,

        "ocr": False,

        "inference_size": INFERENCE_SIZE,

        "startup_warnings": [
            x for x in [
                main_model_error,
                helmet_model_error
            ]
            if x
        ]
    })


# ============================================================
# IMAGE DECODING
# ============================================================

def read_uploaded_image(file):

    data = file.read()

    if not data:

        raise ValueError(
            "Uploaded image is empty."
        )

    if len(data) > MAX_IMAGE_SIZE:

        raise ValueError(
            "Image is larger than 12 MB."
        )

    image = Image.open(
        BytesIO(data)
    ).convert("RGB")

    image = np.array(image)

    return image


# ============================================================
# RESIZE IMAGE
# ============================================================

def prepare_image(image):

    height, width = image.shape[:2]

    max_dimension = 1280

    if max(height, width) > max_dimension:

        scale = (
            max_dimension /
            float(max(height, width))
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    return image


# ============================================================
# MAIN YOLO DETECTION
# ============================================================

def run_main_detection(image):

    detections = []

    if main_model is None:

        return detections

    logger.info(
        "Running main YOLO model..."
    )

    results = main_model.predict(
        source=image,
        conf=MAIN_CONF,
        imgsz=INFERENCE_SIZE,
        device="cpu",
        verbose=False
    )

    if not results:

        return detections

    result = results[0]

    if result.boxes is None:

        return detections

    names = result.names

    boxes = result.boxes

    for i in range(
        len(boxes)
    ):

        cls_id = int(
            boxes.cls[i].item()
        )

        confidence = float(
            boxes.conf[i].item()
        )

        xyxy = boxes.xyxy[i].cpu().numpy()

        x1, y1, x2, y2 = [
            int(v) for v in xyxy
        ]

        class_name = names.get(
            cls_id,
            str(cls_id)
        )

        detections.append({

            "class_id": cls_id,

            "class_name": class_name,

            "confidence": confidence,

            "box": [
                x1,
                y1,
                x2,
                y2
            ]

        })

    return detections


# ============================================================
# HELMET DETECTION
# ============================================================

def run_helmet_detection(image):

    detections = []

    if helmet_model is None:

        return detections

    logger.info(
        "Running helmet model..."
    )

    results = helmet_model.predict(
        source=image,
        conf=HELMET_CONF,
        imgsz=INFERENCE_SIZE,
        device="cpu",
        verbose=False
    )

    if not results:

        return detections

    result = results[0]

    if result.boxes is None:

        return detections

    names = result.names

    boxes = result.boxes

    for i in range(
        len(boxes)
    ):

        cls_id = int(
            boxes.cls[i].item()
        )

        confidence = float(
            boxes.conf[i].item()
        )

        xyxy = boxes.xyxy[i].cpu().numpy()

        x1, y1, x2, y2 = [
            int(v) for v in xyxy
        ]

        class_name = names.get(
            cls_id,
            str(cls_id)
        )

        detections.append({

            "class_id": cls_id,

            "class_name": class_name,

            "confidence": confidence,

            "box": [
                x1,
                y1,
                x2,
                y2
            ]

        })

    return detections


# ============================================================
# CREATE VIOLATIONS
# ============================================================

def build_violations(
    main_detections,
    helmet_detections
):

    violations = []

    # --------------------------------
    # Main model violations
    # --------------------------------

    for detection in main_detections:

        cls_id = detection["class_id"]

        if cls_id in TARGET_CLASSES:

            violations.append({

                "type": TARGET_CLASSES[cls_id],

                "confidence":
                    detection["confidence"],

                "source": "25-Class YOLO",

                "box":
                    detection["box"]

            })


    # --------------------------------
    # Helmet violation
    # --------------------------------

    for detection in helmet_detections:

        if (
            detection["class_id"]
            == HELMET_NO_HELMET_ID
        ):

            violations.append({

                "type": "No Helmet",

                "confidence":
                    detection["confidence"],

                "source": "Helmet Model",

                "box":
                    detection["box"]

            })


    return violations


# ============================================================
# HELMET STATUS
# ============================================================

def get_helmet_status(
    helmet_detections
):

    no_helmet = [
        x for x in helmet_detections
        if x["class_id"]
        == HELMET_NO_HELMET_ID
    ]

    if no_helmet:

        confidence = max(
            x["confidence"]
            for x in no_helmet
        )

        return (
            "No Helmet",
            confidence
        )

    if helmet_detections:

        return (
            "Helmet Detected",
            max(
                x["confidence"]
                for x in helmet_detections
            )
        )

    return (
        "Not detected",
        0.0
    )


# ============================================================
# DECISION ENGINE
# ============================================================

def make_decision(
    violations
):

    if not violations:

        return (
            "NOT_DETECTED",
            0.0
        )


    confidence = max(
        v["confidence"]
        for v in violations
    )


    if confidence >= REVIEW_THRESHOLD:

        return (
            "DETECTED",
            confidence
        )


    return (
        "HUMAN_REVIEW",
        confidence
    )


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

def draw_annotations(
    image,
    main_detections,
    helmet_detections,
    violations
):

    output = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------
    # Main model boxes
    # --------------------------------

    for detection in main_detections:

        x1, y1, x2, y2 = \
            detection["box"]

        cls_id = \
            detection["class_id"]

        confidence = \
            detection["confidence"]

        label = \
            detection["class_name"]


        if cls_id in TARGET_CLASSES:

            label = (
                TARGET_CLASSES[cls_id]
                + " "
                + f"{confidence:.0%}"
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )


    # --------------------------------
    # Helmet boxes
    # --------------------------------

    for detection in helmet_detections:

        x1, y1, x2, y2 = \
            detection["box"]

        cls_id = \
            detection["class_id"]

        confidence = \
            detection["confidence"]

        if cls_id == HELMET_NO_HELMET_ID:

            label = (
                "No Helmet "
                + f"{confidence:.0%}"
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )


    # --------------------------------
    # Encode result
    # --------------------------------

    output_rgb = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        output_rgb
    )

    buffer = BytesIO()

    pil_image.save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


# ============================================================
# ANALYZE API
# ============================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    logger.info("=" * 60)

    logger.info(
        "NEW TRAFFIC IMAGE ANALYSIS"
    )

    logger.info("=" * 60)


    try:

        # --------------------------------
        # Check upload
        # --------------------------------

        if "image" not in request.files:

            return jsonify({
                "error":
                    "No image was uploaded."
            }), 400


        uploaded_file = \
            request.files["image"]


        if uploaded_file.filename == "":

            return jsonify({
                "error":
                    "No image file selected."
            }), 400


        # --------------------------------
        # Read image
        # --------------------------------

        logger.info(
            "Reading uploaded image: %s",
            uploaded_file.filename
        )

        image = read_uploaded_image(
            uploaded_file
        )

        logger.info(
            "Original image shape: %s",
            image.shape
        )


        # --------------------------------
        # Resize
        # --------------------------------

        image = prepare_image(
            image
        )

        logger.info(
            "Prepared image shape: %s",
            image.shape
        )


        # --------------------------------
        # Main detection
        # --------------------------------

        main_detections = \
            run_main_detection(image)


        logger.info(
            "Main detections: %d",
            len(main_detections)
        )


        # --------------------------------
        # Helmet detection
        # --------------------------------

        helmet_detections = \
            run_helmet_detection(image)


        logger.info(
            "Helmet detections: %d",
            len(helmet_detections)
        )


        # --------------------------------
        # Build violations
        # --------------------------------

        violations = \
            build_violations(
                main_detections,
                helmet_detections
            )


        # --------------------------------
        # Helmet status
        # --------------------------------

        helmet_status, \
        helmet_confidence = \
            get_helmet_status(
                helmet_detections
            )


        # --------------------------------
        # Decision
        # --------------------------------

        decision, \
        overall_confidence = \
            make_decision(
                violations
            )


        # --------------------------------
        # Annotated image
        # --------------------------------

        annotated_image = \
            draw_annotations(
                image,
                main_detections,
                helmet_detections,
                violations
            )


        # --------------------------------
        # Response
        # --------------------------------

        response = {

            "success": True,

            "decision": decision,

            "overall_confidence":
                overall_confidence,

            "violations":
                violations,

            "helmet_status":
                helmet_status,

            "helmet_confidence":
                helmet_confidence,

            "plate_text":
                None,

            "plate_confidence":
                None,

            "ocr_confidence":
                None,

            "plate_model":
                False,

            "ocr":
                False,

            "annotated_image":
                annotated_image
        }


        logger.info(
            "Analysis completed: %s",
            decision
        )

        logger.info(
            "Violations detected: %d",
            len(violations)
        )

        logger.info("=" * 60)


        return jsonify(response)


    except Exception as e:

        logger.exception(
            "ANALYSIS FAILED"
        )


        return jsonify({

            "success": False,

            "error":
                "Traffic analysis failed.",

            "details":
                str(e)

        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def too_large(error):

    return jsonify({

        "error":
            "Uploaded file is too large."

    }), 413


@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "error":
            "Endpoint not found."

    }), 404


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv("PORT", "5000")
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
