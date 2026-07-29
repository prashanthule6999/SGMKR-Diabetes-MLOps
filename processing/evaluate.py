# model_evaluation

import os
import json
import pickle
import tarfile
import logging
import argparse
import pandas as pd
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-model-artifacts", type=str, required=True)
    parser.add_argument("--input-test-data", type=str, required=True)
    parser.add_argument("--evaluation-output", type=str, required=True)
    parser.add_argument("--target-column", type=str, required=True)

    return parser.parse_args()


def extract_model_artifacts(input_model_artifacts: str):

    model_path = os.path.join(input_model_artifacts, "model.tar.gz")

    logger.info("Extracting model artifacts...")

    with tarfile.open(model_path) as tar:
        tar.extractall(input_model_artifacts)

    logger.info("Extraction complete.")


def load_artifacts(input_model_artifacts: str) -> Dict[str, Any]:
    """Load the artifacts(model, threshold) from a file."""

    file_path = os.path.join(input_model_artifacts, "model_artifact.pkl")

    try:
        with open(file_path, "rb") as f:
            artifacts = pickle.load(f)

        logger.info('Artifacts loaded from %s', file_path)
        return artifacts
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error(
            'Unexpected error occurred while loading the artifacts: %s', e)
        raise


def evaluate_model(clf: Pipeline, best_threshold: float, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model and return the evaluation metrics."""
    try:
        test_prob = clf.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= best_threshold).astype(int)

        metrics_dict = {
            'test_accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred),
            'auc': roc_auc_score(y_test, test_prob)
        }

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            test_pred
        ).ravel()

        metrics_dict.update({
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        })

        logger.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics, evaluation_output):
    path = os.path.join(evaluation_output, "evaluation.json")

    report = {
        "metrics": {
            "accuracy": {
                "value": metrics["test_accuracy"]
            },
            "precision": {
                "value": metrics["precision"]
            },
            "recall": {
                "value": metrics["recall"]
            },
            "f1": {
                "value": metrics["f1"]
            },
            "auc": {
                "value": metrics["auc"]
            }
        },
        "confusion_matrix": {
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"]
        }
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=4)


def main():

    try:

        args = parse_args()

        extract_model_artifacts(args.input_model_artifacts)

        artifacts = load_artifacts(args.input_model_artifacts)
        model = artifacts["model"]
        best_threshold = artifacts["threshold"]

        test_df = pd.read_csv(os.path.join(args.input_test_data, "test.csv"))
        X_test = test_df.drop(columns=[args.target_column])
        y_test = test_df[args.target_column]

        metrics = evaluate_model(model, best_threshold, X_test, y_test)

        os.makedirs(args.evaluation_output, exist_ok=True)
        save_metrics(metrics, args.evaluation_output)

    except Exception:
        logger.exception(
            'Failed to complete the model evaluation process')
        raise


if __name__ == '__main__':
    main()
