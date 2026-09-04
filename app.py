import os
import re
import base64
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO


# ============================================================
# TRAFFICGUARD AI
# Flask + YOLO Backend
# Render Deployment Version
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

# Maximum upload size: 12 MB
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


# ============================================================
# MODEL PATHS
# ============================================================
# IMPORTANT:
# Both models are in the SAME folder as app.py on GitHub.
# Therefore, DO NOT use /content/ or Google Drive paths.

MAIN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "main_25class_best.pt"
)

HELMET_MODEL_PATH = os.path.join(
    BASE_DIR,
    "helmet_balanced_best.pt"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MAIN_CONF = float(
    os.getenv("MAIN_CONF", "0.10")
)

HELMET_CONF = float(
    os.getenv("HELMET_CONF", "0.10")
)

REVIEW_THRESHOLD = float(
    os.getenv("REVIEW_THRESHOLD", "0.60")
)


# ============================================================
# MODEL STATUS
# ============================================================

main_model = None
helmet_model = None

startup_errors = []


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    global main_model
    global helmet_model

    print("")
    print("=" * 70)
    print("TRAFFICGUARD AI - MODEL RESTORATION")
    print("=" * 70)

    print("")
    print("Main model path:")
    print(MAIN_MODEL_PATH)

    print("Main model exists:")
    print(os.path.isfile(MAIN_MODEL_PATH))

    print("")
    print("Helmet model path:")
    print(HELMET_MODEL_PATH)

    print("Helmet model exists:")
    print(os.path.isfile(HELMET_MODEL_PATH))

    print("")
    print("-" * 70)

    # --------------------------------------------------------
    # Main 25-class model
    # --------------------------------------------------------

    if os.path.isfile(MAIN_MODEL_PATH):

        try:

            print("Loading 25-class YOLO model...")

            main_model = YOLO(
                MAIN_MODEL_PATH
            )

            print("✅ 25-class model loaded")

        except Exception as error:

            message = (
                "Main model loading failed: "
                + str(error)
            )

            startup_errors.append(message)

            print("❌", message)

    else:

        message = (
            "Main model not found: "
            + MAIN_MODEL_PATH
        )

        startup_errors.append(message)

        print("❌", message)

    # --------------------------------------------------------
    # Helmet model
    # --------------------------------------------------------

    if os.path.isfile(HELMET_MODEL_PATH):

        try:

            print("Loading helmet YOLO model...")

            helmet_model = YOLO(
                HELMET_MODEL_PATH
            )

            print("✅ Helmet model loaded")

        except Exception as error:

            message = (
                "Helmet model loading failed: "
                + str(error)
            )

            startup_errors.append(message)

            print("❌", message)

    else:

        message = (
            "Helmet model not found: "
            + HELMET_MODEL_PATH
        )

        startup_errors.append(message)

        print("❌", message)

    print("")
    print("=" * 70)

    if main_model is not None:
        print("Main model : READY")
    else:
        print("Main model : FAILED")

    if helmet_model is not None:
        print("Helmet model: READY")
    else:
        print("Helmet model: FAILED")

    print("=" * 70)
    print("")


# Load models when application starts
load_models()


# ============================================================
# MAIN MODEL VIOLATION CLASSES
# ============================================================
# These are the violation class IDs already used in your
# previous backend.
#
# 3  -> Triple Riding
# 4  -> Phone While Driving
# 9  -> Seatbelt Violation
#
# Helmet model provides:
# 0 -> Helmet
# 1 -> No Helmet
# ============================================================

TARGET_MAIN = {

    3: "Triple Riding",

    4: "Phone While Driving",

    9: "Seatbelt Violation",

}

HELMET_NO_HELMET_ID = 1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )


def result_items(result, model):

    detections = []

    if result.boxes is None:
        return detections

    names = model.names

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        coordinates = box.xyxy[
            0
        ].cpu().numpy()

        x1 = int(coordinates[0])
        y1 = int(coordinates[1])
        x2 = int(coordinates[2])
        y2 = int(coordinates[3])

        if isinstance(names, dict):

            class_name = names.get(
                class_id,
                str(class_id)
            )

        else:

            class_name = names[class_id]

        detections.append({

            "id": class_id,

            "name": class_name,

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
# IMAGE ENCODING
# ============================================================

def encode_image(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            90
        ]
    )

    if not success:
        return None

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


# ============================================================
# ANALYZE IMAGE
# ============================================================

def analyze_image(image):

    if main_model is None:

        raise RuntimeError(
            "25-class YOLO model is not loaded."
        )

    if helmet_model is None:

        raise RuntimeError(
            "Helmet YOLO model is not loaded."
        )

    violations = []

    all_boxes = []


    # ========================================================
    # 1. MAIN 25-CLASS YOLO MODEL
    # ========================================================

    print("Running 25-class model...")

    main_results = main_model.predict(

        source=image,

        conf=MAIN_CONF,

        imgsz=640,

        verbose=False

    )

    main_result = main_results[0]

    main_items = result_items(
        main_result,
        main_model
    )


    # --------------------------------------------------------
    # Process main detections
    # --------------------------------------------------------

    for item in main_items:

        all_boxes.append(
            (
                "main",
                item
            )
        )

        class_id = item["id"]

        if class_id in TARGET_MAIN:

            violations.append({

                "type":
                    TARGET_MAIN[class_id],

                "confidence":
                    item["confidence"]

            })


    # ========================================================
    # 2. HELMET MODEL
    # ========================================================

    print("Running helmet model...")

    helmet_results = helmet_model.predict(

        source=image,

        conf=HELMET_CONF,

        imgsz=640,

        verbose=False

    )

    helmet_result = helmet_results[0]

    helmet_items = result_items(
        helmet_result,
        helmet_model
    )


    helmet_status = "Not detected"


    # --------------------------------------------------------
    # Process helmet detections
    # --------------------------------------------------------

    for item in helmet_items:

        all_boxes.append(
            (
                "helmet",
                item
            )
        )

        class_id = item["id"]

        class_name = str(
            item["name"]
        ).lower()


        # No helmet
        if class_id == HELMET_NO_HELMET_ID:

            violations.append({

                "type":
                    "No Helmet",

                "confidence":
                    item["confidence"]

            })

            helmet_status = "No Helmet"


        # Helmet detected
        elif (
            class_id == 0
            or "helmet" in class_name
        ):

            if helmet_status != "No Helmet":

                helmet_status = (
                    "Helmet detected"
                )


    # ========================================================
    # DRAW ANNOTATIONS
    # ========================================================

    annotated = image.copy()


    for source, item in all_boxes:

        x1, y1, x2, y2 = item["box"]


        # --------------------------------------------
        # Red = violation
        # Green = normal detection
        # --------------------------------------------

        is_violation = False

        if item["id"] in TARGET_MAIN:

            is_violation = True

        if (
            source == "helmet"
            and item["id"]
            == HELMET_NO_HELMET_ID
        ):

            is_violation = True


        if is_violation:

            # BGR = red
            color = (
                60,
                70,
                255
            )

        else:

            # BGR = green
            color = (
                80,
                210,
                150
            )


        # --------------------------------------------
        # Bounding box
        # --------------------------------------------

        cv2.rectangle(

            annotated,

            (x1, y1),

            (x2, y2),

            color,

            2

        )


        # --------------------------------------------
        # Label
        # --------------------------------------------

        label = (

            f'{item["name"]} '
            f'{item["confidence"]:.0%}'

        )


        label_width = max(
            120,
            len(label) * 8
        )


        label_y1 = max(
            0,
            y1 - 25
        )


        cv2.rectangle(

            annotated,

            (
                x1,
                label_y1
            ),

            (
                x1 + label_width,
                y1
            ),

            color,

            -1

        )


        cv2.putText(

            annotated,

            label,

            (
                x1 + 5,
                y1 - 7
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                10,
                10,
                15
            ),

            2

        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence_scores = [

        item["confidence"]

        for item in violations

    ]


    if confidence_scores:

        overall_confidence = (

            sum(confidence_scores)
            /
            len(confidence_scores)

        )

    else:

        overall_confidence = 0.0


    # ========================================================
    # DECISION
    # ========================================================

    if not violations:

        decision = "NOT_DETECTED"

    elif (
        overall_confidence
        < REVIEW_THRESHOLD
    ):

        decision = "HUMAN_REVIEW"

    else:

        decision = "DETECTED"


    # ========================================================
    # ENCODE ANNOTATED IMAGE
    # ========================================================

    encoded = encode_image(
        annotated
    )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "success": True,

        "decision":
            decision,

        "violations":
            violations,

        "helmet_status":
            helmet_status,

        "plate_text":
            None,

        "plate_confidence":
            None,

        "ocr_confidence":
            None,

        "overall_confidence":
            overall_confidence,

        "camera_location":
            "Camera 01",

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "annotated_image":
            encoded

    }


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    return send_from_directory(

        BASE_DIR,

        "index.html"

    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return jsonify({

        "status":
            "online",

        "service":
            "TrafficGuard AI",

        "main_model":
            main_model is not None,

        "helmet_model":
            helmet_model is not None,

        "startup_warnings":
            startup_errors

    })


# ============================================================
# ANALYZE API
# ============================================================

@app.post("/api/analyze")
def api_analyze():

    # --------------------------------------------------------
    # Check uploaded image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No image uploaded."

        }), 400


    uploaded = request.files[
        "image"
    ]


    if uploaded.filename == "":

        return jsonify({

            "success": False,

            "error":
                "No image selected."

        }), 400


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    data = uploaded.read()


    if not data:

        return jsonify({

            "success": False,

            "error":
                "Uploaded image is empty."

        }), 400


    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    array = np.frombuffer(

        data,

        dtype=np.uint8

    )


    image = cv2.imdecode(

        array,

        cv2.IMREAD_COLOR

    )


    if image is None:

        return jsonify({

            "success": False,

            "error":
                "Could not decode the image."

        }), 400


    # --------------------------------------------------------
    # Run AI analysis
    # --------------------------------------------------------

    try:

        print("")
        print("=" * 70)
        print("NEW TRAFFIC IMAGE ANALYSIS")
        print("=" * 70)

        result = analyze_image(
            image
        )

        print(
            "Violations:",
            len(
                result["violations"]
            )
        )

        print(
            "Helmet:",
            result["helmet_status"]
        )

        print(
            "Confidence:",
            result[
                "overall_confidence"
            ]
        )

        print("=" * 70)

        return jsonify(result)


    except Exception as error:

        print("")
        print("❌ DETECTION ERROR")
        print(str(error))

        return jsonify({

            "success": False,

            "error":
                f"Detection request failed: {error}"

        }), 500


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def too_large(error):

    return jsonify({

        "success": False,

        "error":
            "Image is too large. Maximum size is 12 MB."

    }), 413


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            "10000"
        )

    )

    print("")
    print("=" * 70)
    print("TRAFFICGUARD AI SERVER")
    print("=" * 70)

    print(
        "Port:",
        port
    )

    print(
        "Main model:",
        os.path.basename(
            MAIN_MODEL_PATH
        )
    )

    print(
        "Helmet model:",
        os.path.basename(
            HELMET_MODEL_PATH
        )
    )

    print("=" * 70)
    print("")

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )
