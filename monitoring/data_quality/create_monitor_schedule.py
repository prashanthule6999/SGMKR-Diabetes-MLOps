import logging
import sagemaker

from config import *
from sagemaker.model_monitor import (
    DefaultModelMonitor,
    CronExpressionGenerator,
)


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
# Baseline location
# --------------------------------------------------

baseline_statistics = (
    f"{BASELINE_OUTPUT_S3_URI}"
    "statistics.json"
)

baseline_constraints = (
    f"{BASELINE_OUTPUT_S3_URI}"
    "constraints.json"
)


# --------------------------------------------------
# Create Monitoring Schedule
# --------------------------------------------------

logger.info(
    "Creating Model Monitor schedule..."
)

logger.info(
    "Endpoint: %s",
    ENDPOINT_NAME,
)

logger.info(
    "Schedule: %s",
    MONITOR_SCHEDULE_NAME,
)


monitor.create_monitoring_schedule(

    endpoint_input=ENDPOINT_NAME,

    output_s3_uri=MONITORING_OUTPUT_S3_URI,

    statistics=baseline_statistics,

    constraints=baseline_constraints,

    schedule_cron_expression=(
        CronExpressionGenerator.hourly()
    ),

    monitor_schedule_name=MONITOR_SCHEDULE_NAME,

    enable_cloudwatch_metrics=True,
)


logger.info(
    "Model Monitor schedule created successfully."
)