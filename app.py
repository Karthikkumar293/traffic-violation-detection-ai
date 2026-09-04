import os
import base64
from datetime import datetime

# ============================================================
# RENDER MEMORY / CPU SETTINGS
# ============================================================

# Keep CPU thread usage low on small Render instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

try:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

except Exception:
    torch = None


import cv2
import numpy as np

from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO


# ============================================================
# TRAFFICGUARD AI
# Flask + YOLO Backend
# Render Optimized Version
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

# Maximum upload size
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


# ============================================================
# MODEL PATHS
# ============================================================
#
# Your GitHub repository contains:
#
# main_25class_best.pt
# helmet_balanced_best.pt
#
# They are beside app.py.
#
# DO NOT use:
# /content/
# /content/drive/
# Google Drive paths
#
# ============================================================

MAIN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "main_25class_best.pt"
)

HELMET_MODEL_PATH = os.path.join(
    BASE_DIR,
    "helmet_balanced_best.pt"
)


# ============================================================
# SETTINGS
# ============================================================

# Keep the confidence threshold low because your Colab
# pipeline used 0.10.
MAIN_CONF = float(
    os.getenv("MAIN_CONF", "0.10")
)

HELMET_CONF = float(
    os.getenv("HELMET_CONF", "0.10")
)

# Cases below this confidence go to human review.
REVIEW_THRESHOLD = float(
    os.getenv("REVIEW_THRESHOLD", "0.60")
)

# Render CPU optimization.
# 416 is considerably lighter than the original 640 inference.
INFERENCE_SIZE = int(
    os.getenv("INFERENCE_SIZE", "416")
)

# Maximum detections returned by each model.
MAX_DETECTIONS = int(
    os.getenv("MAX_DETECTIONS", "50")
)


# ============================================================
# TARGET CLASSES
# ============================================================

# Main 25-class YOLO model
#
# 3 -> Triple Riding
# 4 -> Phone While Driving
# 9 -> Seatbelt Violation
#
TARGET_MAIN = {
    3: "Triple Riding",
    4: "Phone While Driving",
    9: "Seatbelt Violation"
}

# Helmet model
#
# 0 -> Helmet
# 1 -> No Helmet
#
HELMET_ID = 0
HELMET_NO_HELMET_ID = 1


# ============================================================
# MODEL VARIABLES
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
    print("TRAFFICGUARD AI - MODEL LOADING")
    print("=" * 70)

    print("")
    print("Base directory:")
    print(BASE_DIR)

    print("")
    print("Main model:")
    print(MAIN_MODEL_PATH)

    print("Main model exists:")
    print(os.path.isfile(MAIN_MODEL_PATH))

    print("")
    print("Helmet model:")
    print(HELMET_MODEL_PATH)

    print("Helmet model exists:")
    print(os.path.isfile(HELMET_MODEL_PATH))

    # --------------------------------------------------------
    # MAIN MODEL
    # --------------------------------------------------------

    if not os.path.isfile(MAIN_MODEL_PATH):

        error = (
            "Main model not found: "
            + MAIN_MODEL_PATH
        )

        startup_errors.append(error)

        print("❌", error)

    else:

        try:

            print("")
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

    # --------------------------------------------------------
    # HELMET MODEL
    # --------------------------------------------------------

    if not os.path.isfile(HELMET_MODEL_PATH):

        error = (
            "Helmet model not found: "
            + HELMET_MODEL_PATH
        )

        startup_errors.append(error)

        print("❌", error)

    else:

        try:

            print("")
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

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("")
    print("=" * 70)

    print(
        "Main model:",
        "READY" if main_model is not None else "FAILED"
    )

    print(
        "Helmet model:",
        "READY" if helmet_model is not None else "FAILED"
    )

    print("=" * 70)
    print("")


# Load models exactly once when Gunicorn starts.
load_models()


# ============================================================
# RESULT CONVERSION
# ============================================================

def get_class_name(model, class_id):

    names = model.names

    if isinstance(names, dict):

        return names.get(
            class_id,
            str(class_id)
        )

    try:

        return names[class_id]

    except Exception:

        return str(class_id)


def extract_detections(result, model):

    detections = []

    if result is None:
        return detections

    if result.boxes is None:
        return detections

    for box in result.boxes:

        try:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            coords = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
            )

            x1 = max(
                0,
                int(coords[0])
            )

            y1 = max(
                0,
                int(coords[1])
            )

            x2 = max(
                x1,
                int(coords[2])
            )

            y2 = max(
                y1,
                int(coords[3])
            )

            class_name = get_class_name(
                model,
                class_id
            )

            detections.append({

                "id": class_id,

                "name": str(
                    class_name
                ),

                "confidence": confidence,

                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ]

            })

        except Exception as error:

            print(
                "Detection parsing warning:",
                repr(error)
            )

    return detections


# ============================================================
# IMAGE ENCODING
# ============================================================

def encode_image(image):

    try:

        # Keep returned image reasonably small.
        height, width = image.shape[:2]

        max_width = 1200

        if width > max_width:

            scale = (
                max_width /
                float(width)
            )

            new_width = max_width

            new_height = int(
                height * scale
            )

            image = cv2.resize(
                image,
                (
                    new_width,
                    new_height
                ),
                interpolation=cv2.INTER_AREA
            )

        success, buffer = cv2.imencode(
            ".jpg",
            image,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                82
            ]
        )

        if not success:
            return None

        return base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

    except Exception as error:

        print(
            "Image encoding error:",
            repr(error)
        )

        return None


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detection(
    image,
    detection,
    is_violation
):

    x1, y1, x2, y2 = detection["box"]

    if is_violation:

        # Red
        color = (
            60,
            70,
            255
        )

    else:

        # Green
        color = (
            80,
            210,
            150
        )

    # Bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    label = (
        f'{detection["name"]} '
        f'{detection["confidence"]:.0%}'
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.55

    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    label_width = max(
        120,
        text_width + 10
    )

    label_height = (
        text_height +
        baseline +
        8
    )

    label_y2 = max(
        label_height,
        y1
    )

    label_y1 = max(
        0,
        label_y2 - label_height
    )

    # Label background
    cv2.rectangle(
        image,
        (
            x1,
            label_y1
        ),
        (
            x1 + label_width,
            label_y2
        ),
        color,
        -1
    )

    # Label text
    text_y = (
        label_y2 -
        baseline -
        4
    )

    cv2.putText(
        image,
        label,
        (
            x1 + 5,
            text_y
        ),
        font,
        font_scale,
        (
            10,
            10,
            15
        ),
        thickness,
        cv2.LINE_AA
    )


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

    # --------------------------------------------------------
    # Optional memory cleanup
    # --------------------------------------------------------

    if torch is not None:

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    violations = []

    all_detections = []

    # ========================================================
    # STEP 1
    # MAIN 25-CLASS YOLO
    # ========================================================

    print("")
    print("Running 25-class model...")

    main_results = main_model.predict(

        source=image,

        conf=MAIN_CONF,

        iou=0.50,

        imgsz=INFERENCE_SIZE,

        max_det=MAX_DETECTIONS,

        device="cpu",

        verbose=False

    )

    main_result = main_results[0]

    main_detections = extract_detections(
        main_result,
        main_model
    )

    print(
        "Main detections:",
        len(main_detections)
    )

    # Process target violations
    for detection in main_detections:

        class_id = detection["id"]

        if class_id in TARGET_MAIN:

            violations.append({

                "type":
                    TARGET_MAIN[class_id],

                "confidence":
                    detection["confidence"],

                "box":
                    detection["box"]

            })

            all_detections.append({

                "detection":
                    detection,

                "violation":
                    True

            })

        else:

            # Keep other detected classes visible
            # but don't count them as target violations.
            all_detections.append({

                "detection":
                    detection,

                "violation":
                    False

            })

    # ========================================================
    # STEP 2
    # HELMET MODEL
    # ========================================================

    print("")
    print("Running helmet model...")

    helmet_results = helmet_model.predict(

        source=image,

        conf=HELMET_CONF,

        iou=0.50,

        imgsz=INFERENCE_SIZE,

        max_det=MAX_DETECTIONS,

        device="cpu",

        verbose=False

    )

    helmet_result = helmet_results[0]

    helmet_detections = extract_detections(
        helmet_result,
        helmet_model
    )

    print(
        "Helmet detections:",
        len(helmet_detections)
    )

    helmet_status = "Not detected"

    no_helmet_found = False
    helmet_found = False

    for detection in helmet_detections:

        class_id = detection["id"]

        class_name = (
            str(
                detection["name"]
            ).lower()
        )

        # ----------------------------------------------------
        # NO HELMET
        # ----------------------------------------------------

        if (
            class_id ==
            HELMET_NO_HELMET_ID
        ):

            no_helmet_found = True

            violations.append({

                "type":
                    "No Helmet",

                "confidence":
                    detection["confidence"],

                "box":
                    detection["box"]

            })

            all_detections.append({

                "detection":
                    detection,

                "violation":
                    True

            })

        # ----------------------------------------------------
        # HELMET
        # ----------------------------------------------------

        elif (
            class_id == HELMET_ID
            or "helmet" in class_name
        ):

            helmet_found = True

            all_detections.append({

                "detection":
                    detection,

                "violation":
                    False

            })

    # Helmet status
    if no_helmet_found:

        helmet_status = "No Helmet"

    elif helmet_found:

        helmet_status = "Helmet detected"

    else:

        helmet_status = "Not detected"

    # ========================================================
    # STEP 3
    # REMOVE DUPLICATE VIOLATION TYPES
    # ========================================================

    # If multiple detections of the same target are returned,
    # keep the highest-confidence one for the dashboard.

    best_violations = {}

    for violation in violations:

        violation_type = (
            violation["type"]
        )

        confidence = (
            violation["confidence"]
        )

        if (
            violation_type
            not in best_violations
        ):

            best_violations[
                violation_type
            ] = violation

        elif confidence > best_violations[
            violation_type
        ]["confidence"]:

            best_violations[
                violation_type
            ] = violation

    violations = list(
        best_violations.values()
    )

    # ========================================================
    # STEP 4
    # CONFIDENCE
    # ========================================================

    confidence_scores = [

        float(
            violation["confidence"]
        )

        for violation in violations

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
    # STEP 5
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
    # STEP 6
    # DRAW RESULT
    # ========================================================

    annotated = image.copy()

    for item in all_detections:

        detection = item[
            "detection"
        ]

        is_violation = item[
            "violation"
        ]

        draw_detection(
            annotated,
            detection,
            is_violation
        )

    # ========================================================
    # STEP 7
    # ENCODE IMAGE
    # ========================================================

    encoded_image = encode_image(
        annotated
    )

    # ========================================================
    # NUMBER PLATE
    # ========================================================
    #
    # No plate_best.pt is currently deployed.
    # Therefore don't pretend OCR is working.
    #
    # The frontend will correctly display:
    # Number Plate -> Not detected
    #
    # ========================================================

    plate_text = None
    plate_confidence = None
    ocr_confidence = None

    # ========================================================
    # FINAL RESULT
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
            plate_text,

        "plate_confidence":
            plate_confidence,

        "ocr_confidence":
            ocr_confidence,

        "overall_confidence":
            float(
                overall_confidence
            ),

        "camera_location":
            "Camera 01",

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "annotated_image":
            encoded_image
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

        "plate_model":
            False,

        "ocr":
            False,

        "inference_size":
            INFERENCE_SIZE,

        "startup_warnings":
            startup_errors
    })


# ============================================================
# ANALYZE API
# ============================================================

@app.post("/api/analyze")
def api_analyze():

    # --------------------------------------------------------
    # CHECK FILE
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
    # READ FILE
    # --------------------------------------------------------

    data = uploaded.read()

    if not data:

        return jsonify({

            "success": False,

            "error":
                "Uploaded image is empty."

        }), 400

    # --------------------------------------------------------
    # DECODE IMAGE
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
    # LIMIT VERY LARGE INPUT IMAGES
    # --------------------------------------------------------

    height, width = image.shape[:2]

    max_dimension = 1600

    if max(
        height,
        width
    ) > max_dimension:

        scale = (
            max_dimension /
            float(
                max(
                    height,
                    width
                )
            )
        )

        new_width = max(
            1,
            int(
                width * scale
            )
        )

        new_height = max(
            1,
            int(
                height * scale
            )
        )

        image = cv2.resize(

            image,

            (
                new_width,
                new_height
            ),

            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------------
    # RUN AI
    # --------------------------------------------------------

    try:

        print("")
        print("=" * 70)
        print("NEW TRAFFIC IMAGE ANALYSIS")
        print("=" * 70)

        print(
            "Input size:",
            image.shape[1],
            "x",
            image.shape[0]
        )

        result = analyze_image(
            image
        )

        print("")
        print(
            "Decision:",
            result["decision"]
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
        print("")

        return jsonify(
            result
        )

    except Exception as error:

        print("")
        print("=" * 70)
        print("❌ AI ANALYSIS ERROR")
        print("=" * 70)
        print(
            repr(error)
        )
        print("=" * 70)
        print("")

        return jsonify({

            "success": False,

            "error":
                "AI analysis failed.",

            "details":
                str(error)

        }), 500


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error":
            "Image is too large. Maximum size is 12 MB."

    }), 413


# ============================================================
# GENERAL ERROR HANDLER
# ============================================================

@app.errorhandler(Exception)
def handle_exception(error):

    print(
        "Unhandled application error:",
        repr(error)
    )

    return jsonify({

        "success": False,

        "error":
            "Server error.",

        "details":
            str(error)

    }), 500


# ============================================================
# LOCAL SERVER
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

    print(
        "Inference size:",
        INFERENCE_SIZE
    )

    print("=" * 70)
    print("")

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False
    )
