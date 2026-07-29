# data preprocessing

import os
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def replace_invalid_zeros(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = df[cols].replace(0, np.nan)
    return df


def train_val_test_split(df: pd.DataFrame, test_size: float, val_size: float, random_state: int, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Validation split is used for threshold optimization,
    # preventing the test set from influencing threshold selection.
    train_full, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state
    )

    # Split training portion again
    train, val = train_test_split(
        train_full,
        test_size=val_size,   # 0.25 × 0.80 = 0.20 (20% of original dataset)
        stratify=train_full[target_column],
        random_state=random_state
    )

    return train, val, test


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-data", type=str, required=True,
                        help="Input directory mounted by SageMaker")
    parser.add_argument("--input-file", type=str,
                        required=True, help="CSV filename")
    parser.add_argument("--train-output", type=str, required=True,
                        help="Output directory mounted by SageMaker")
    parser.add_argument("--validation-output", type=str,
                        required=True, help="Output directory mounted by SageMaker")
    parser.add_argument("--test-output", type=str, required=True,
                        help="Output directory mounted by SageMaker")
    parser.add_argument("--test-size", type=float, required=True)
    parser.add_argument("--val-size", type=float, required=True)
    parser.add_argument("--random-state", type=int, required=True)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--columns", type=str, required=True,
                        help="Comma separated column names having invalid zero values")

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    try:

        input_file = os.path.join(args.input_data, args.input_file)
        logger.info("Reading dataset from %s", input_file)

        df = pd.read_csv(input_file)
        logger.info(f"Original dataset shape: {df.shape}")

        columns = [col.strip() for col in args.columns.split(",")]

        # Transform the data
        cleaned_df = replace_invalid_zeros(df, columns)

        train_df, val_df, test_df = train_val_test_split(
            df=cleaned_df,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            target_column=args.target_column,
        )

        os.makedirs(args.train_output, exist_ok=True)
        os.makedirs(args.validation_output, exist_ok=True)
        os.makedirs(args.test_output, exist_ok=True)

        train_df.to_csv(
            os.path.join(args.train_output, "train.csv"),
            index=False,
        )

        val_df.to_csv(
            os.path.join(args.validation_output, "validation.csv"),
            index=False,
        )

        test_df.to_csv(
            os.path.join(args.test_output, "test.csv"),
            index=False,
        )

        logger.info("Train saved to %s", args.train_output)
        logger.info("Validation saved to %s", args.validation_output)
        logger.info("Test saved to %s", args.test_output)

        logger.info("Train shape: %s", train_df.shape)
        logger.info("Validation shape: %s", val_df.shape)
        logger.info("Test shape: %s", test_df.shape)

        logger.info("Data preprocessing completed successfully.")

    except Exception as e:

        logger.exception("Data preprocessing failed.")

        raise


if __name__ == '__main__':
    main()
