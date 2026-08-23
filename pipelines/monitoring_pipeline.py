import sagemaker

from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.steps import (
    ProcessingStep,
)

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

    base_job_name="diabetes-drift",

    sagemaker_session=session,
)


# --------------------------------------------------
# Data Drift Processing Step
# --------------------------------------------------

drift_step = ProcessingStep(

    name="DataDriftDetection",

    processor=processor,

    code="processing/detect_data_drift.py",

    inputs=[

        ProcessingInput(

            source=TRAINING_DATA_S3_URI,

            destination=
                "/opt/ml/processing/reference",
        ),

        ProcessingInput(

            source=PRODUCTION_DATA_S3_URI,

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
)


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

pipeline = Pipeline(

    name="DiabetesMonitoringPipeline",

    steps=[
        drift_step
    ],

    sagemaker_session=session,
)


# --------------------------------------------------
# Create / Update Pipeline
# --------------------------------------------------

pipeline.upsert(

    role_arn=
        EXECUTION_ROLE_ARN
)


print(
    "Monitoring pipeline created successfully."
)