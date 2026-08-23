# Baseline represents the data distribution the model was trained on
import logging
import sagemaker

from config import *

from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# SageMaker Session
# --------------------------------------------------

sagemaker_session = sagemaker.Session()

# --------------------------------------------------
# Create Model Monitor
# --------------------------------------------------

monitor = DefaultModelMonitor(

    role=EXECUTION_ROLE_ARN,

    instance_count=1,

    instance_type="ml.m5.large",

    volume_size_in_gb=20,

    max_runtime_in_seconds=3600,

    sagemaker_session=sagemaker_session,
)


# --------------------------------------------------
# Create Baseline
# --------------------------------------------------

logger.info(
    "Creating Model Monitor baseline..."
)

logger.info(
    "Training data: %s",
    TRAINING_DATA_S3_URI,
)

logger.info(
    "Baseline output: %s",
    BASELINE_OUTPUT_S3_URI,
)


monitor.suggest_baseline(

    baseline_dataset=TRAINING_DATA_S3_URI,

    dataset_format=DatasetFormat.csv(
        header=True
    ),

    output_s3_uri=BASELINE_OUTPUT_S3_URI,

    wait=True,

    logs=True,
)


logger.info(
    "Baseline creation job submitted successfully."
)
