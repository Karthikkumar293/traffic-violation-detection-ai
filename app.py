from pathlib import Path
import logging

from flask import Flask, jsonify
from ultralytics import YOLO


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("TrafficGuard")


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MAIN_MODEL_NAME = "main_25class_best.pt"
HELMET_MODEL_NAME = "helmet_balanced_best.pt"


# ============================================================
# FIND MODEL
# ============================================================

def find_model(filename):

    possible_paths = [
        BASE_DIR / filename,
        BASE_DIR / "models" / filename,
    ]

    for path in possible_paths:

        if path.is_file():

            logger.info(
                "Model found: %s",
                path
            )

            return str(path)

    logger.error(
        "Model not found: %s",
        filename
    )

    return None


# ============================================================
# MODEL LOADING
# ============================================================

logger.info(
    "============================================================"
)

logger.info(
    "TRAFFICGUARD AI MODEL LOADING"
)

logger.info(
    "============================================================"
)


MAIN_MODEL_PATH = find_model(
    MAIN_MODEL_NAME
)

HELMET_MODEL_PATH = find_model(
    HELMET_MODEL_NAME
)


main_model = None
helmet_model = None


# ============================================================
# LOAD MAIN MODEL
# ============================================================

if MAIN_MODEL_PATH:

    try:

        logger.info(
            "Loading main 25-class model..."
        )

        main_model = YOLO(
            MAIN_MODEL_PATH
        )

        logger.info(
            "Main model loaded successfully"
        )

    except Exception as e:

        logger.exception(
            "Failed to load main model: %s",
            e
        )

else:

    logger.error(
        "Main model is unavailable"
    )


# ============================================================
# LOAD HELMET MODEL
# ============================================================

if HELMET_MODEL_PATH:

    try:

        logger.info(
            "Loading helmet model..."
        )

        helmet_model = YOLO(
            HELMET_MODEL_PATH
        )

        logger.info(
            "Helmet model loaded successfully"
        )

    except Exception as e:

        logger.exception(
            "Failed to load helmet model: %s",
            e
        )

else:

    logger.error(
        "Helmet model is unavailable"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "service": "TrafficGuard AI",
        "status": "online",
        "main_model": main_model is not None,
        "helmet_model": helmet_model is not None,
        "main_model_path": MAIN_MODEL_PATH,
        "helmet_model_path": HELMET_MODEL_PATH
    })


# ============================================================
# ROOT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return """
    <h1>TrafficGuard AI</h1>
    <p>Service is online.</p>
    <p>API: /api/analyze</p>
    <p>Health: /api/health</p>
    """


# ============================================================
# FINAL STATUS
# ============================================================

logger.info(
    "============================================================"
)

logger.info(
    "MODEL RESTORATION"
)

logger.info(
    "============================================================"
)

logger.info(
    "Main model : %s",
    main_model is not None
)

logger.info(
    "Helmet model: %s",
    helmet_model is not None
)

logger.info(
    "============================================================"
)
