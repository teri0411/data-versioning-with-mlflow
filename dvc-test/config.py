"""Configuration for the wine quality prediction project."""

import os
import logging

# Paths
DATA_PATH = "data/wine-quality.csv"
MODEL_PATH = "models/model.pt"

# MLflow settings
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "wine-quality-prediction"

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT
    )
    return logging.getLogger(__name__)
