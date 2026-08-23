# This file is the orchestrator for Data Drift Processing Job
# It creates a SageMaker Processing Job and tells SageMaker where to get the data, 
# which script to run, and where to put the result.
import sagemaker

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
)

from sagemaker.sklearn.processing import (
    SKLearnProcessor,
)

from config import (
    EXECUTION_ROLE_ARN,
    TRAINING_DATA_S3_URI,
    PRODUCTION_DATA_S3_URI,
    DRIFT_OUTPUT_S3_URI,
)


# --------------------------------------------------
# SageMaker Session
# --------------------------------------------------

session = sagemaker.Session()


# --------------------------------------------------
# Processor
# --------------------------------------------------

processor = SKLearnProcessor(

    framework_version="1.2-1",

    role=EXECUTION_ROLE_ARN,

    instance_type="ml.m5.large",

    instance_count=1,

    base_job_name=
        "diabetes-drift",

    sagemaker_session=session,
)


# --------------------------------------------------
# Run Processing Job
# --------------------------------------------------

processor.run(

    code=
        "processing/detect_data_drift.py",

    inputs=[

        ProcessingInput(

            source=
                TRAINING_DATA_S3_URI,

            destination=
                "/opt/ml/processing/reference",
        ),

        ProcessingInput(

            source=
                PRODUCTION_DATA_S3_URI,

            destination=
                "/opt/ml/processing/production",
        ),
    ],

    outputs=[

        ProcessingOutput(

            source=
                "/opt/ml/processing/output",

            destination=
                DRIFT_OUTPUT_S3_URI,
        ),
    ],

    wait=True,

    logs=True,
)