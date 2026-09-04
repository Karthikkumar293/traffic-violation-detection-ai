from pathlib import Path
from ultralytics import YOLO
import logging

logger = logging.getLogger("TrafficGuard")

# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MAIN_MODEL_NAME = "main_25class_best.pt"
HELMET_MODEL_NAME = "helmet_balanced_best.pt"


def find_model(filename):
    """
    Find model whether it is:
      1. In repository root
      2. In models/ folder
    """

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
# LOAD MAIN MODEL
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
# MAIN YOLO MODEL
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
# HELMET YOLO MODEL
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
# FINAL MODEL STATUS
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
