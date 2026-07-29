import os
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--penalty", type=str, required=True)
    parser.add_argument("--solver", type=str, required=True)
    parser.add_argument("--C", type=float, required=True)
    parser.add_argument("--class-weight", type=str, required=True)
    parser.add_argument("--random-state", type=int, required=True)

    return parser.parse_args()


def load_dataset(channel: str) -> pd.DataFrame:
    channel_path = os.environ[f"SM_CHANNEL_{channel.upper()}"]
    path = os.path.join(channel_path, f"{channel}.csv")

    # In the backend it construct like below in container
    # SM_CHANNEL_TRAIN=/opt/ml/input/data/train/train.csv
    # SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation/validation.csv

    logger.info(f"Loading {path}")
    return pd.read_csv(path)


def train_model(X, y, penalty, solver, C, class_weight, random_state):

    model = LogisticRegression(
        penalty=penalty,
        solver=solver,
        C=C,
        class_weight=class_weight,
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    return pipeline


def find_best_threshold(model, X_val, y_val):

    probs = model.predict_proba(X_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_val,
        probs,
    )

    idx = np.argmax(tpr - fpr)

    return float(thresholds[idx])


def main():

    args = parse_args()

    class_weight = (
        None
        if args.class_weight == "None"
        else args.class_weight
    )

    train_df = load_dataset("train")
    val_df = load_dataset("validation")

    X_train = train_df.drop(columns=[args.target_column])
    y_train = train_df[args.target_column]

    logger.info("Training initial model...")

    model = train_model(
        X_train,
        y_train,
        args.penalty,
        args.solver,
        args.C,
        class_weight,
        args.random_state,
    )

    logger.info("Finding optimal threshold...")

    X_val = val_df.drop(columns=[args.target_column])
    y_val = val_df[args.target_column]

    threshold = find_best_threshold(
        model,
        X_val,
        y_val,
    )

    logger.info(f"Best threshold = {threshold:.4f}")

    logger.info("Retraining using Train + Validation...")

    final_df = pd.concat(
        [train_df, val_df],
        ignore_index=True,
    )

    X_final = final_df.drop(columns=[args.target_column])
    y_final = final_df[args.target_column]

    final_model = train_model(
        X_final,
        y_final,
        args.penalty,
        args.solver,
        args.C,
        class_weight,
        args.random_state,
    )

    model_dir = os.environ["SM_MODEL_DIR"]

    os.makedirs(model_dir, exist_ok=True)

    artifacts = {
        "model": final_model,
        "threshold": threshold,
    }

    with open(os.path.join(model_dir, "model_artifact.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
