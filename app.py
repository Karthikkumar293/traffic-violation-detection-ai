import os
import base64
import cv2
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO


# ============================================================
# TRAFFICGUARD AI
# Flask Backend
# ============================================================

app = Flask(__name__, static_folder=".", static_url_path="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# MODEL PATHS
# ============================================================

MAIN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "main_25class_best.pt"
)

HELMET_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "helmet_balanced_best.pt"
)

# Optional plate model
PLATE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "plate_best.pt"
)


# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

DETECTION_CONFIDENCE = 0.40
HUMAN_REVIEW_THRESHOLD = 0.60


# ============================================================
# FOUR MAIN VIOLATIONS
#
# Based on your model logic:
#
# 3 = Triple Riding
# 4 = Phone While Driving
# 9 = Seatbelt Violation
#
# No Helmet is handled by the separate helmet model.
# ============================================================

TRIPLE_RIDING_ID = 3
PHONE_VIOLATION_ID = 4
SEATBELT_VIOLATION_ID = 9

VIOLATION_NAMES = {
    TRIPLE_RIDING_ID: "Triple Riding",
    PHONE_VIOLATION_ID: "Phone While Driving",
    SEATBELT_VIOLATION_ID: "Seatbelt Violation",
}


# ============================================================
# LOAD MODELS
# ============================================================

main_model = None
helmet_model = None
plate_model = None


def load_models():

    global main_model
    global helmet_model
    global plate_model

    print("=" * 70)
    print("TRAFFICGUARD AI - MODEL INITIALIZATION")
    print("=" * 70)

    # ----------------------------
    # Main 25-class model
    # ----------------------------

    if os.path.isfile(MAIN_MODEL_PATH):

        try:
            main_model = YOLO(MAIN_MODEL_PATH)

            print(
                "25-Class YOLO Model: READY"
            )
            print(
                MAIN_MODEL_PATH
            )

        except Exception as e:

            print(
                "25-Class Model Load Error:",
                e
            )

    else:

        print(
            "25-Class Model NOT FOUND:"
        )
        print(
            MAIN_MODEL_PATH
        )

    # ----------------------------
    # Helmet model
    # ----------------------------

    if os.path.isfile(HELMET_MODEL_PATH):

        try:
            helmet_model = YOLO(
                HELMET_MODEL_PATH
            )

            print(
                "Helmet Model: READY"
            )
            print(
                HELMET_MODEL_PATH
            )

        except Exception as e:

            print(
                "Helmet Model Load Error:",
                e
            )

    else:

        print(
            "Helmet Model NOT FOUND:"
        )
        print(
            HELMET_MODEL_PATH
        )

    # ----------------------------
    # Plate model
    # ----------------------------

    if os.path.isfile(PLATE_MODEL_PATH):

        try:

            plate_model = YOLO(
                PLATE_MODEL_PATH
            )

            print(
                "Plate Model: READY"
            )

        except Exception as e:

            print(
                "Plate Model Load Error:",
                e
            )

    else:

        print(
            "Plate Model: NOT AVAILABLE"
        )

    print("=" * 70)


load_models()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def confidence_percent(value):

    value = float(value)

    if value <= 1:
        value *= 100

    return round(value, 2)


def violation_name(value):

    return (
        str(value)
        .replace("_", " ")
        .title()
    )


def image_to_base64(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image
    )

    if not success:
        return None

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


# ============================================================
# MAIN MODEL ANALYSIS
# ============================================================

def analyze_main_model(image):

    violations = []

    if main_model is None:

        return violations

    results = main_model.predict(
        source=image,
        conf=0.10,
        iou=0.50,
        imgsz=640,
        max_det=100,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:

        return violations

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        if class_id in VIOLATION_NAMES:

            violations.append({

                "type":
                    VIOLATION_NAMES[class_id],

                "confidence":
                    confidence,

                "box": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]

            })

    return violations


# ============================================================
# HELMET MODEL ANALYSIS
# ============================================================

def analyze_helmet_model(image):

    violations = []

    if helmet_model is None:

        return violations

    results = helmet_model.predict(
        source=image,
        conf=0.10,
        iou=0.50,
        imgsz=640,
        max_det=100,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:

        return violations

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        # According to your model logic:
        # class 1 = WITHOUT HELMET

        if class_id == 1:

            violations.append({

                "type":
                    "No Helmet",

                "confidence":
                    confidence,

                "box": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ]

            })

    return violations


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    violations
):

    output = image.copy()

    for violation in violations:

        box = violation["box"]

        x1, y1, x2, y2 = box

        confidence = confidence_percent(
            violation["confidence"]
        )

        label = (
            f"{violation['type']} "
            f"{confidence:.1f}%"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            2
        )

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        text_y = max(
            y1 - 10,
            text_height + 5
        )

        cv2.rectangle(
            output,
            (
                x1,
                text_y - text_height - 8
            ),
            (
                x1 + text_width + 8,
                text_y + baseline
            ),
            (0, 165, 255),
            -1
        )

        cv2.putText(
            output,
            label,
            (
                x1 + 4,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


# ============================================================
# PLATE DETECTION
# ============================================================

def analyze_plate(image):

    if plate_model is None:

        return {

            "plate_text": None,

            "plate_confidence": None,

            "ocr_confidence": None

        }

    try:

        results = plate_model.predict(
            source=image,
            conf=0.25,
            iou=0.50,
            imgsz=640,
            max_det=20,
            verbose=False
        )

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            return {

                "plate_text": None,

                "plate_confidence": None,

                "ocr_confidence": None

            }

        # Select highest-confidence plate

        best_box = None
        best_conf = 0

        for box in result.boxes:

            confidence = float(
                box.conf[0].item()
            )

            if confidence > best_conf:

                best_conf = confidence
                best_box = box

        if best_box is None:

            return {

                "plate_text": None,

                "plate_confidence": None,

                "ocr_confidence": None

            }

        x1, y1, x2, y2 = (
            best_box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(
            image.shape[1],
            x2
        )

        y2 = min(
            image.shape[0],
            y2
        )

        plate_crop = image[
            y1:y2,
            x1:x2
        ]

        if plate_crop.size == 0:

            return {

                "plate_text": None,

                "plate_confidence":
                    best_conf,

                "ocr_confidence": None

            }

        # ------------------------------------------------
        # OCR
        #
        # PaddleOCR can be connected here.
        # The backend intentionally does not crash
        # when OCR is unavailable.
        # ------------------------------------------------

        plate_text = None
        ocr_confidence = None

        try:

            from paddleocr import PaddleOCR

            ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False
            )

            ocr_result = ocr.ocr(
                plate_crop,
                cls=True
            )

            if (
                ocr_result
                and ocr_result[0]
            ):

                texts = []

                scores = []

                for item in ocr_result[0]:

                    if len(item) >= 2:

                        text_data = item[1]

                        if (
                            isinstance(
                                text_data,
                                (list, tuple)
                            )
                            and len(text_data) >= 2
                        ):

                            text = str(
                                text_data[0]
                            )

                            score = float(
                                text_data[1]
                            )

                            texts.append(text)
                            scores.append(score)

                if texts:

                    plate_text = "".join(
                        texts
                    )

                    if scores:

                        ocr_confidence = (
                            sum(scores)
                            / len(scores)
                        )

        except Exception as ocr_error:

            print(
                "OCR unavailable:",
                ocr_error
            )

        return {

            "plate_text":
                plate_text,

            "plate_confidence":
                best_conf,

            "ocr_confidence":
                ocr_confidence

        }

    except Exception as e:

        print(
            "Plate analysis error:",
            e
        )

        return {

            "plate_text": None,

            "plate_confidence": None,

            "ocr_confidence": None

        }


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

    confidences = [

        float(v["confidence"])

        for v in violations

    ]

    overall = max(
        confidences
    )

    if overall < HUMAN_REVIEW_THRESHOLD:

        return (
            "HUMAN_REVIEW",
            overall
        )

    return (
        "DETECTED",
        overall
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# ============================================================
# HEALTH API
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "online",

        "main_model":
            main_model is not None,

        "helmet_model":
            helmet_model is not None,

        "plate_model":
            plate_model is not None,

        "ocr": True

    })


# ============================================================
# ANALYSIS API
# ============================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    if "image" not in request.files:

        return jsonify({

            "error":
                "No image uploaded."

        }), 400

    uploaded_file = request.files[
        "image"
    ]

    if uploaded_file.filename == "":

        return jsonify({

            "error":
                "No image selected."

        }), 400

    try:

        file_bytes = (
            uploaded_file
            .read()
        )

        image_array = np.frombuffer(
            file_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({

                "error":
                    "Invalid image file."

            }), 400

        # ====================================================
        # STEP 1 — MAIN 25-CLASS YOLO
        # ====================================================

        main_violations = (
            analyze_main_model(
                image
            )
        )

        # ====================================================
        # STEP 2 — HELMET MODEL
        # ====================================================

        helmet_violations = (
            analyze_helmet_model(
                image
            )
        )

        # Combine violation results

        violations = (
            main_violations
            +
            helmet_violations
        )

        # ====================================================
        # HELMET STATUS
        # ====================================================

        if helmet_violations:

            helmet_status = (
                "No Helmet"
            )

        elif helmet_model is not None:

            helmet_status = (
                "Helmet Detected"
            )

        else:

            helmet_status = (
                "Unavailable"
            )

        # ====================================================
        # STEP 3 — NUMBER PLATE + OCR
        # ====================================================

        plate_data = analyze_plate(
            image
        )

        # ====================================================
        # STEP 4 — DECISION
        # ====================================================

        decision, overall_confidence = (
            make_decision(
                violations
            )
        )

        # ====================================================
        # DRAW RESULT
        # ====================================================

        annotated_image = (
            draw_detections(
                image,
                violations
            )
        )

        encoded_image = (
            image_to_base64(
                annotated_image
            )
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "decision":
                decision,

            "overall_confidence":
                overall_confidence,

            "violations": [

                {

                    "type":
                        v["type"],

                    "confidence":
                        v["confidence"],

                    "box":
                        v["box"]

                }

                for v in violations

            ],

            "helmet_status":
                helmet_status,

            "plate_text":
                plate_data[
                    "plate_text"
                ],

            "plate_confidence":
                plate_data[
                    "plate_confidence"
                ],

            "ocr_confidence":
                plate_data[
                    "ocr_confidence"
                ],

            "annotated_image":
                encoded_image

        })

    except Exception as e:

        print(
            "ANALYSIS ERROR:",
            repr(e)
        )

        return jsonify({

            "error":
                "AI analysis failed.",

            "details":
                str(e)

        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
