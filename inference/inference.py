# we write inference.py when we want to customize any part of the inference process.
# this file know how to serve the predictions
import os
import json
import pickle
import logging
import pandas as pd
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def model_fn(model_dir: str) -> dict[str, Any]:
    """
    Load the trained model artifact when the SageMaker endpoint starts.
    SageMaker automatically downloads and extracts model.tar.gz into:
        /opt/ml/model/
    This function is called only once when the inference container starts.
    Args:
        model_dir:
            Directory containing the extracted model artifacts.

    Returns:
        Dictionary containing:
            {
                "model": final_model,
                "threshold": threshold,
            }
    """

    try:

        model_path = os.path.join(
            model_dir,
            "model_artifact.pkl"
        )

        logger.info(
            "Loading model artifact from %s",
            model_path,
        )

        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)

        logger.info("Model artifact loaded successfully.")

        logger.info(
            "Stored threshold: %.6f",
            artifacts["threshold"],
        )

        return artifacts
    except Exception:
        logger.exception("model loading failed.")
        raise


def input_fn(request_body, request_content_type):

    try:
        logger.info(
            "Received request with content type: %s",
            request_content_type,
        )

        if request_content_type != "application/json":
            raise ValueError(
                f"Unsupported content type: {request_content_type}"
            )

        user_input = json.loads(request_body)

        input_df = pd.DataFrame([user_input])

        input_df = input_df.astype({
            "Pregnancies": "int64",
            "Glucose": "float64",
            "BloodPressure": "float64",
            "SkinThickness": "float64",
            "Insulin": "float64",
            "BMI": "float64",
            "DiabetesPedigreeFunction": "float64",
            "Age": "int64",
        })

        logger.info("Input converted to DataFrame.")

        return input_df

    except Exception:
        logger.exception("Failed to parse request.")
        raise


def predict_fn(input_df, model_artifact):
    """
    Perform prediction using the trained model and the
    optimized decision threshold.
    """
    try:
        model = model_artifact["model"]
        threshold = model_artifact["threshold"]

        # Predict probabilities
        probabilities = model.predict_proba(input_df)[0]

        # Probability of positive class (Diabetes)
        probability = float(probabilities[1])

        # Apply custom threshold
        prediction = int(probability >= threshold)

        logger.info(
            "Prediction=%d Probability=%.4f Threshold=%.4f",
            prediction,
            probability,
            threshold,
        )

        return {
            "prediction": prediction,
            "diabetes_probability": round(probability, 4),
            "threshold": round(threshold, 4),
            "class_probabilities": {
                "No Diabetes": round(float(probabilities[0]), 4),
                "Diabetes": round(float(probabilities[1]), 4)
            }
        }
    except Exception:
        logger.exception("Prediction failed.")
        raise


def output_fn(prediction, accept):
    """
    Convert prediction result into the HTTP response.
    """

    if accept not in (None, "*/*", "application/json"):
        raise ValueError(
            f"Unsupported accept type: {accept}"
        )

    return json.dumps(prediction), "application/json"
