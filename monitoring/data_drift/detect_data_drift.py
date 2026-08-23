# drift calculation
import os
import json
import logging

import numpy as np
import pandas as pd


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# SageMaker Processing paths
# --------------------------------------------------

REFERENCE_DIR = (
    "/opt/ml/processing/reference"
)

PRODUCTION_DIR = (
    "/opt/ml/processing/production"
)

OUTPUT_DIR = (
    "/opt/ml/processing/output"
)


# --------------------------------------------------
# Features
# --------------------------------------------------

FEATURES = [

    "pregnancies",

    "glucose",

    "blood_pressure",

    "skin_thickness",

    "insulin",

    "bmi",

    "diabetes_pedigree_function",

    "age",
]


# --------------------------------------------------
# Find CSV
# --------------------------------------------------

def find_csv(directory):

    files = [

        file

        for file in os.listdir(directory)

        if file.endswith(".csv")
    ]

    if not files:

        raise FileNotFoundError(
            f"No CSV file found in {directory}"
        )

    return os.path.join(
        directory,
        files[0],
    )


# --------------------------------------------------
# Calculate PSI
# --------------------------------------------------

def calculate_psi(
    reference,
    production,
    bins=10,
):

    # Create bins from reference data

    breakpoints = np.percentile(

        reference,

        np.linspace(
            0,
            100,
            bins + 1,
        ),
    )

    breakpoints = np.unique(
        breakpoints
    )

    if len(breakpoints) < 3:

        return 0.0

    reference_counts, _ = np.histogram(

        reference,

        bins=breakpoints,
    )

    production_counts, _ = np.histogram(

        production,

        bins=breakpoints,
    )

    reference_percentage = (

        reference_counts /
        len(reference)
    )

    production_percentage = (

        production_counts /
        len(production)
    )

    # Prevent division by zero

    reference_percentage = np.where(

        reference_percentage == 0,

        0.0001,

        reference_percentage,
    )

    production_percentage = np.where(

        production_percentage == 0,

        0.0001,

        production_percentage,
    )

    psi = np.sum(

        (
            production_percentage
            -
            reference_percentage
        )

        *

        np.log(

            production_percentage
            /
            reference_percentage
        )
    )

    return float(psi)


# --------------------------------------------------
# Classify Drift
# --------------------------------------------------

def classify_drift(psi):

    if psi < 0.10:

        return "NO_DRIFT"

    elif psi < 0.25:

        return "MODERATE_DRIFT"

    else:

        return "SIGNIFICANT_DRIFT"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    logger.info(
        "Loading reference data..."
    )

    reference_file = find_csv(
        REFERENCE_DIR
    )

    logger.info(
        "Reference file: %s",
        reference_file,
    )

    logger.info(
        "Loading production data..."
    )

    production_file = find_csv(
        PRODUCTION_DIR
    )

    logger.info(
        "Production file: %s",
        production_file,
    )

    reference_data = pd.read_csv(
        reference_file
    )

    production_data = pd.read_csv(
        production_file
    )

    results = []

    # --------------------------------------------------
    # Calculate drift per feature
    # --------------------------------------------------

    for feature in FEATURES:

        logger.info(
            "Checking feature: %s",
            feature,
        )

        reference_values = (

            reference_data[feature]

            .dropna()

            .values
        )

        production_values = (

            production_data[feature]

            .dropna()

            .values
        )

        psi = calculate_psi(

            reference_values,

            production_values,
        )

        status = classify_drift(
            psi
        )

        results.append({

            "feature": feature,

            "psi": psi,

            "status": status,
        })

    # --------------------------------------------------
    # Overall drift
    # --------------------------------------------------

    drift_detected = any(

        result["status"]
        == "SIGNIFICANT_DRIFT"

        for result in results
    )

    report = {

        "drift_detected":
            drift_detected,

        "features":
            results,
    }

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    os.makedirs(

        OUTPUT_DIR,

        exist_ok=True,
    )

    output_file = os.path.join(

        OUTPUT_DIR,

        "drift_report.json",
    )

    with open(

        output_file,

        "w",
    ) as file:

        json.dump(

            report,

            file,

            indent=4,
        )

    logger.info(
        "Drift report created: %s",
        output_file,
    )


if __name__ == "__main__":

    main()
