import os
import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO


# ============================================================
# APP
# ============================================================

app = Flask(__name__)

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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MAIN_MODEL = BASE_DIR / "main_25class_best.pt"
HELMET_MODEL = BASE_DIR / "helmet_balanced_best.pt"


# ============================================================
# FIND MODEL
# ============================================================

def find_model(filename):

    locations = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
    ]

    for path in locations:

        if path.exists() and path.is_file():

            logger.info("MODEL FOUND: %s", path)

            return path

    logger.error("MODEL NOT FOUND: %s", filename)

    return None


main_model_path = find_model("main_25class_best.pt")
helmet_model_path = find_model("helmet_balanced_best.pt")


# ============================================================
# LOAD MODELS
# ============================================================

main_model = None
helmet_model = None


logger.info("=" * 60)
logger.info("TRAFFICGUARD AI MODEL LOADING")
logger.info("=" * 60)


if main_model_path:

    try:

        logger.info("Loading main model...")

        main_model = YOLO(
            str(main_model_path)
        )

        logger.info(
            "MAIN MODEL LOADED SUCCESSFULLY"
        )

    except Exception:

        logger.exception(
            "MAIN MODEL LOAD FAILED"
        )


if helmet_model_path:

    try:

        logger.info("Loading helmet model...")

        helmet_model = YOLO(
            str(helmet_model_path)
        )

        logger.info(
            "HELMET MODEL LOADED SUCCESSFULLY"
        )

    except Exception:

        logger.exception(
            "HELMET MODEL LOAD FAILED"
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
# YOLO DETECTION
# ============================================================

def detect(model, image, confidence=0.25):

    detections = []

    if model is None:

        return detections

    results = model.predict(
        source=image,
        imgsz=416,
        conf=confidence,
        verbose=False,
        device="cpu"
    )

    if not results:

        return detections

    result = results[0]

    if result.boxes is None:

        return detections

    boxes = result.boxes

    for i in range(len(boxes)):

        try:

            class_id = int(
                boxes.cls[i].item()
            )

            confidence_value = float(
                boxes.conf[i].item()
            )

            coordinates = (
                boxes.xyxy[i]
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
                    confidence_value,
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
                "Detection parsing error: %s",
                error
            )

    return detections


# ============================================================
# DRAW BOXES
# ============================================================

def draw_boxes(image, detections):

    output = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        name = detection["class_name"]

        confidence = detection["confidence"]

        label = (
            f"{name} "
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
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output


# ============================================================
# IMAGE TO BASE64
# ============================================================

def image_to_base64(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )

    if not success:

        return None

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# GET UPLOADED IMAGE
# ============================================================

def get_uploaded_image():

    # --------------------------------------------------------
    # METHOD 1: NORMAL MULTIPART FILE
    # --------------------------------------------------------

    if request.files:

        logger.info(
            "FILES RECEIVED: %s",
            list(request.files.keys())
        )

        for field_name in request.files:

            uploaded = request.files[field_name]

            if uploaded and uploaded.filename:

                logger.info(
                    "Using uploaded field: %s",
                    field_name
                )

                data = uploaded.read()

                if data:

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
    # METHOD 2: BASE64 JSON
    # --------------------------------------------------------

    try:

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

                if isinstance(value, str):

                    if "," in value:

                        value = value.split(
                            ",",
                            1
                        )[1]

                    try:

                        data = base64.b64decode(
                            value
                        )

                        array = np.frombuffer(
                            data,
                            dtype=np.uint8
                        )

                        image = cv2.imdecode(
                            array,
                            cv2.IMREAD_COLOR
                        )

                        if image is not None:

                            logger.info(
                                "Base64 image received from: %s",
                                key
                            )

                            return image

                    except Exception:

                        continue

    except Exception:

        pass


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

        "ocr": False,

        "plate_model": False,

        "inference_size": 416,

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

    logger.info(
        "NEW ANALYSIS REQUEST"
    )

    logger.info(
        "Content-Type: %s",
        request.content_type
    )

    logger.info(
        "Request files: %s",
        list(request.files.keys())
    )

    logger.info("=" * 60)


    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if main_model is None:

        return jsonify({

            "success": False,

            "status": "Analysis failed",

            "error": (
                "Main model is not loaded on Render."
            )

        }), 503


    # --------------------------------------------------------
    # GET IMAGE
    # --------------------------------------------------------

    image = get_uploaded_image()


    if image is None:

        logger.error(
            "NO IMAGE COULD BE READ FROM REQUEST"
        )

        return jsonify({

            "success": False,

            "status": "Analysis failed",

            "error": (
                "No readable image was uploaded."
            ),

            "debug_files": list(
                request.files.keys()
            )

        }), 400


    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    height, width = image.shape[:2]

    logger.info(
        "IMAGE RECEIVED: %sx%s",
        width,
        height
    )


    # --------------------------------------------------------
    # MAIN MODEL
    # --------------------------------------------------------

    logger.info(
        "Running main YOLO model..."
    )

    main_detections = detect(
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
            "Running helmet YOLO model..."
        )

        helmet_detections = detect(
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

    annotated = draw_boxes(
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

        highest_confidence = max(
            confidence_values
        )

    else:

        overall_confidence = 0.0

        highest_confidence = 0.0


    # --------------------------------------------------------
    # VIOLATIONS
    # --------------------------------------------------------

    violations = []


    for detection in helmet_detections:

        name = (
            detection["class_name"]
            .lower()
            .replace("-", "_")
        )


        if (
            "no_helmet" in name
            or "nohelmet" in name
            or "without_helmet" in name
            or "withouthelmet" in name
        ):

            violations.append({

                "type": "No Helmet",

                "confidence": (
                    detection["confidence"]
                ),

                "detection": detection

            })


    # --------------------------------------------------------
    # VIOLATION TYPE
    # --------------------------------------------------------

    if violations:

        violation_type = (
            violations[0]["type"]
        )

        analysis_status = (
            "Violation Detected"
        )

    elif all_detections:

        violation_type = "-"

        analysis_status = (
            "Analysis Complete"
        )

    else:

        violation_type = "-"

        analysis_status = (
            "No Detection"
        )


    # --------------------------------------------------------
    # NUMBER PLATE
    # --------------------------------------------------------

    number_plate = "Not detected"

    plate_confidence = None


    for detection in main_detections:

        name = (
            detection["class_name"]
            .lower()
        )


        if (
            "plate" in name
            or "license" in name
            or "number_plate" in name
        ):

            number_plate = (
                detection["class_name"]
            )

            plate_confidence = (
                detection["confidence"]
            )

            break


    # --------------------------------------------------------
    # RESULT IMAGE
    # --------------------------------------------------------

    result_image = image_to_base64(
        annotated
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

    response = {

        "success": True,

        "status": analysis_status,

        "analysis_status": analysis_status,

        "overall_confidence": round(
            overall_confidence,
            4
        ),

        "highest_confidence": round(
            highest_confidence,
            4
        ),

        "violation_type": violation_type,

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

        "annotated_image": result_image,

        "result_image": result_image,

        "image": result_image,

        "image_url": result_image,

        "width": width,

        "height": height

    }


    logger.info(
        "ANALYSIS COMPLETED SUCCESSFULLY"
    )

    logger.info(
        "Violations: %d",
        len(violations)
    )

    logger.info(
        "Total detections: %d",
        len(all_detections)
    )


    return jsonify(response)


# ============================================================
# ROOT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    index_path = BASE_DIR / "index.html"

    if index_path.exists():

        return send_from_directory(
            BASE_DIR,
            "index.html"
        )

    return jsonify({

        "service": "TrafficGuard AI",

        "status": "online",

        "message": "index.html not found",

        "health": "/api/health",

        "analyze": "/api/analyze"

    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def too_large(error):

    return jsonify({

        "success": False,

        "status": "Analysis failed",

        "error": (
            "Image is too large. "
            "Maximum size is 15 MB."
        )

    }), 413


@app.errorhandler(Exception)
def server_error(error):

    logger.exception(
        "UNHANDLED SERVER ERROR"
    )

    return jsonify({

        "success": False,

        "status": "Analysis failed",

        "error": str(error)

    }), 500


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
