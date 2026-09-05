import os
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("TrafficGuard")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MAIN_MODEL_NAME = "main_25class_best.pt"
HELMET_MODEL_NAME = "helmet_balanced_best.pt"


# ============================================================
# FIND MODEL
# ============================================================

def find_model(filename):

    paths = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
    ]

    for path in paths:

        if path.is_file():

            logger.info(
                "Model found: %s",
                path
            )

            return path

    logger.error(
        "Model NOT found: %s",
        filename
    )

    return None


# ============================================================
# LOAD MODELS
# ============================================================

logger.info("=" * 60)
logger.info("TRAFFICGUARD AI MODEL LOADING")
logger.info("=" * 60)


main_model_path = find_model(
    MAIN_MODEL_NAME
)

helmet_model_path = find_model(
    HELMET_MODEL_NAME
)


main_model = None
helmet_model = None


# ============================================================
# MAIN MODEL
# ============================================================

if main_model_path:

    try:

        logger.info(
            "Loading main model..."
        )

        main_model = YOLO(
            str(main_model_path)
        )

        logger.info(
            "Main model loaded successfully"
        )

    except Exception:

        logger.exception(
            "Main model loading failed"
        )


# ============================================================
# HELMET MODEL
# ============================================================

if helmet_model_path:

    try:

        logger.info(
            "Loading helmet model..."
        )

        helmet_model = YOLO(
            str(helmet_model_path)
        )

        logger.info(
            "Helmet model loaded successfully"
        )

    except Exception:

        logger.exception(
            "Helmet model loading failed"
        )


logger.info("=" * 60)
logger.info("MODEL STATUS")
logger.info("=" * 60)

logger.info(
    "Main model : %s",
    main_model is not None
)

logger.info(
    "Helmet model: %s",
    helmet_model is not None
)

logger.info("=" * 60)


# ============================================================
# CLASS NAME
# ============================================================

def get_class_name(model, class_id):

    try:

        names = model.names

        if isinstance(names, dict):

            return str(
                names.get(
                    class_id,
                    f"class_{class_id}"
                )
            )

        if isinstance(names, list):

            if 0 <= class_id < len(names):

                return str(
                    names[class_id]
                )

    except Exception:

        pass

    return f"class_{class_id}"


# ============================================================
# YOLO INFERENCE
# ============================================================

def run_detection(model, image):

    detections = []

    if model is None:

        return detections

    try:

        results = model.predict(
            source=image,
            imgsz=320,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        if not results:

            return detections

        result = results[0]

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

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                x1, y1, x2, y2 = [
                    int(round(x))
                    for x in coordinates
                ]

                class_name = get_class_name(
                    model,
                    class_id
                )

                detections.append({

                    "class_id": class_id,

                    "class_name": class_name,

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]

                })

            except Exception as error:

                logger.warning(
                    "Could not parse detection: %s",
                    error
                )

    except Exception as error:

        logger.exception(
            "YOLO inference failed: %s",
            error
        )

        raise

    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    detections
):

    output = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        class_name = (
            detection["class_name"]
        )

        confidence = (
            detection["confidence"]
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_data_url(image):

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            80
        ]
    )

    if not success:

        return None

    encoded_data = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded_data
    )


# ============================================================
# GET IMAGE FROM REQUEST
# ============================================================

def get_request_image():

    # --------------------------------------------------------
    # MULTIPART FILE
    # --------------------------------------------------------

    if request.files:

        logger.info(
            "Received file fields: %s",
            list(request.files.keys())
        )

        for field_name in request.files:

            uploaded_file = (
                request.files[field_name]
            )

            if (
                uploaded_file
                and uploaded_file.filename
            ):

                logger.info(
                    "Using uploaded file field: %s",
                    field_name
                )

                data = (
                    uploaded_file.read()
                )

                if not data:

                    continue

                array = np.frombuffer(
                    data,
                    dtype=np.uint8
                )

                image = cv2.imdecode(
                    array,
                    cv2.IMREAD_COLOR
                )

                if image is not None:

                    return image


    # --------------------------------------------------------
    # BASE64 JSON
    # --------------------------------------------------------

    body = request.get_json(
        silent=True
    )

    if body:

        possible_keys = [
            "image",
            "file",
            "photo",
            "upload",
            "image_data",
            "base64",
            "data"
        ]

        for key in possible_keys:

            value = body.get(key)

            if not value:

                continue

            if not isinstance(
                value,
                str
            ):

                continue

            try:

                if "," in value:

                    value = value.split(
                        ",",
                        1
                    )[1]

                decoded = base64.b64decode(
                    value
                )

                array = np.frombuffer(
                    decoded,
                    dtype=np.uint8
                )

                image = cv2.imdecode(
                    array,
                    cv2.IMREAD_COLOR
                )

                if image is not None:

                    logger.info(
                        "Base64 image received: %s",
                        key
                    )

                    return image

            except Exception:

                continue


    return None


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "service": "TrafficGuard AI",

        "status": "online",

        "main_model": (
            main_model is not None
        ),

        "helmet_model": (
            helmet_model is not None
        ),

        "main_model_path": (
            str(main_model_path)
            if main_model_path
            else None
        ),

        "helmet_model_path": (
            str(helmet_model_path)
            if helmet_model_path
            else None
        ),

        "inference_size": 320,

        "ocr": False,

        "plate_model": False,

        "startup_warnings": []

    })


# ============================================================
# ANALYZE
# ============================================================

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def analyze():

    logger.info("=" * 60)
    logger.info("NEW ANALYSIS REQUEST")
    logger.info(
        "Content-Type: %s",
        request.content_type
    )
    logger.info(
        "File fields: %s",
        list(request.files.keys())
    )
    logger.info("=" * 60)


    # --------------------------------------------------------
    # MAIN MODEL CHECK
    # --------------------------------------------------------

    if main_model is None:

        return jsonify({

            "success": False,

            "status": "Analysis failed",

            "error": (
                "Main YOLO model is not loaded."
            )

        }), 503


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = get_request_image()


    if image is None:

        return jsonify({

            "success": False,

            "status": "Analysis failed",

            "error": (
                "No readable image was uploaded."
            ),

            "received_fields": list(
                request.files.keys()
            )

        }), 400


    height, width = image.shape[:2]


    logger.info(
        "Image received: %sx%s",
        width,
        height
    )


    # --------------------------------------------------------
    # MAIN MODEL
    # --------------------------------------------------------

    logger.info(
        "Running main model..."
    )

    main_detections = run_detection(
        main_model,
        image
    )


    logger.info(
        "Main detections: %d",
        len(main_detections)
    )


    # --------------------------------------------------------
    # HELMET MODEL
    # --------------------------------------------------------

    helmet_detections = []


    if helmet_model is not None:

        logger.info(
            "Running helmet model..."
        )

        helmet_detections = run_detection(
            helmet_model,
            image
        )

        logger.info(
            "Helmet detections: %d",
            len(helmet_detections)
        )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    all_detections = (
        main_detections
        + helmet_detections
    )


    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    annotated_image = draw_detections(
        image,
        all_detections
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence_values = [

        item["confidence"]

        for item in all_detections

    ]


    if confidence_values:

        overall_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        overall_confidence = 0.0


    # --------------------------------------------------------
    # BASIC VIOLATION CHECK
    # --------------------------------------------------------

    violations = []


    for detection in helmet_detections:

        class_name = (
            detection["class_name"]
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )


        if (
            "no_helmet" in class_name
            or "nohelmet" in class_name
            or "without_helmet" in class_name
            or "withouthelmet" in class_name
        ):

            violations.append({

                "type": "No Helmet",

                "confidence": (
                    detection["confidence"]
                ),

                "detection": detection

            })


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if violations:

        analysis_status = (
            "Violation Detected"
        )

        violation_type = (
            violations[0]["type"]
        )

    elif all_detections:

        analysis_status = (
            "Analysis Complete"
        )

        violation_type = "-"

    else:

        analysis_status = (
            "No Detection"
        )

        violation_type = "-"


    # --------------------------------------------------------
    # NUMBER PLATE
    # --------------------------------------------------------

    number_plate = "Not detected"

    plate_confidence = None


    for detection in main_detections:

        class_name = (
            detection["class_name"]
            .lower()
        )

        if (
            "plate" in class_name
            or "license" in class_name
            or "number_plate" in class_name
        ):

            number_plate = (
                detection["class_name"]
            )

            plate_confidence = (
                detection["confidence"]
            )

            break


    # --------------------------------------------------------
    # ENCODE RESULT
    # --------------------------------------------------------

    result_image = image_to_data_url(
        annotated_image
    )


    if result_image is None:

        return jsonify({

            "success": False,

            "status": "Analysis failed",

            "error": (
                "Could not create result image."
            )

        }), 500


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return jsonify({

        "success": True,

        "status": analysis_status,

        "analysis_status": analysis_status,

        "overall_confidence": round(
            overall_confidence,
            4
        ),

        "violation_type": (
            violation_type
        ),

        "number_plate": (
            number_plate
        ),

        "ocr_confidence": None,

        "plate_confidence": (
            plate_confidence
        ),

        "detected_violations": len(
            violations
        ),

        "violations": violations,

        "detections": all_detections,

        "main_detections": (
            main_detections
        ),

        "helmet_detections": (
            helmet_detections
        ),

        "annotated_image": (
            result_image
        ),

        "result_image": (
            result_image
        ),

        "image": (
            result_image
        ),

        "image_url": (
            result_image
        ),

        "width": width,

        "height": height

    })


# ============================================================
# ROOT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    index_file = (
        BASE_DIR / "index.html"
    )

    if index_file.exists():

        return send_from_directory(
            BASE_DIR,
            "index.html"
        )

    return """
    <h1>TrafficGuard AI</h1>
    <p>Service is online.</p>
    <p>index.html is missing.</p>
    """


# ============================================================
# UPLOAD ERROR
# ============================================================

@app.errorhandler(413)
def upload_too_large(error):

    return jsonify({

        "success": False,

        "status": "Analysis failed",

        "error": (
            "Image is too large. "
            "Maximum allowed size is 10 MB."
        )

    }), 413


# ============================================================
# START
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
