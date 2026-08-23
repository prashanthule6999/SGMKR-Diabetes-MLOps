BUCKET = "my-diabetes"

PROJECT_NAME = "DiabetesPrediction"

MODEL_PACKAGE_GROUP_NAME = "DiabetesPredictionModel"

MODEL_NAME_PREFIX = "diabetes-model"

ENDPOINT_NAME = "diabetes-endpoint"

INSTANCE_TYPE = "ml.t2.medium"

INITIAL_INSTANCE_COUNT = 1

AWS_REGION = "us-east-1"

EXECUTION_ROLE_ARN = "arn:aws:iam::<account-id>:role/<your-sagemaker-execution-role>"

GITHUB_OWNER = "PrashantHule"

GITHUB_REPO = "Diabetes-MLOps"

WORKFLOW_FILE = "deploy.yml"

BRANCH = "main"

# ============================================================
# MODEL MONITORING
# ============================================================
# S3 location where SageMaker captures production
# inference requests and responses.
PRODUCTION_MONITORING_DATA_S3_URI = (
    f"s3://{BUCKET}/monitoring/endpoint-data/")

# Percentage of production requests to capture.
DATA_CAPTURE_SAMPLING_PERCENTAGE = 100

# Baseline
TRAINING_DATA_S3_URI = (f"s3://{BUCKET}/processing/train/train.csv")

# S3 location where Model Monitor baseline
# statistics and constraints will be stored.
BASELINE_OUTPUT_S3_URI = (f"s3://{BUCKET}/monitoring/baseline")

BASELINE_STATISTICS_S3_URI = (
    f"s3://{BUCKET}/monitoring/baseline/statistics.json")

BASELINE_CONSTRAINTS_S3_URI = (
    f"s3://{BUCKET}/monitoring/baseline/constraints.json")

# S3 location where Model Monitor execution
# reports will be stored.
MONITORING_OUTPUT_S3_URI = (f"s3://{BUCKET}/monitoring/results/")

MONITOR_SCHEDULE_NAME = ("diabetes-data-quality-monitor")

CLOUD_WATCH_ALARM_NAME = ("diabetes-model-monitor-alarm")

SNS_TOPIC_NAME = ("diabetes-monitor-alerts")

SNS_TOPIC_ARN = (
    "arn:aws:sns:ap-south-1:123456789012:"
    "diabetes-monitor-alerts"
)

ALERT_EMAIL = "your-email@example.com"

MODEL_QUALITY_DATA_S3_URI = (
    f"s3://{BUCKET}/model-quality/data/"
)


# --------------------------------------------------
# Data Drift
# --------------------------------------------------

TRAINING_DATA_S3_URI = (
    "s3://my-diabetes/training/train.csv"
)

PRODUCTION_DATA_S3_URI = (
    "s3://my-diabetes/monitoring/"
    "production-test/production.csv"
)

DRIFT_OUTPUT_S3_URI = (
    "s3://my-diabetes/monitoring/"
    "drift/reports/"
)


# ============================================================
# SageMaker Deployment Configuration
# ============================================================

# ------------------------------------------------------------
# Deployment Strategy
# ------------------------------------------------------------

# Current strategy:
#   Blue/Green deployment with Canary traffic shifting
#
# Supported by SageMaker:
#   CANARY
#   LINEAR
#   ALL_AT_ONCE
#
DEPLOYMENT_STRATEGY = "CANARY"


# ------------------------------------------------------------
# Canary Configuration
# ------------------------------------------------------------

# Percentage of the new (green) fleet to use during
# the initial canary phase.
#
# Example:
#   10 = 10% of green fleet capacity
#
# SageMaker requires the canary size to be <= 50%.
#
CANARY_SIZE_PERCENT = 10


# ------------------------------------------------------------
# Deployment Wait Configuration
# ------------------------------------------------------------

# Time to wait between traffic-shifting steps.
#
# For CANARY:
#   This is the canary baking period before
#   remaining traffic moves to the new fleet.
#
CANARY_WAIT_INTERVAL_SECONDS = 300


# Time to keep the old (blue) fleet alive after
# the new fleet has received 100% traffic.
#
TERMINATION_WAIT_SECONDS = 300


# Maximum time allowed for the entire deployment.
#
# Must be greater than the relevant waiting periods.
#
MAXIMUM_EXECUTION_TIMEOUT_SECONDS = 1800


# ------------------------------------------------------------
# Automatic Rollback
# ------------------------------------------------------------

# CloudWatch alarm used by SageMaker to automatically
# rollback the deployment if the alarm enters ALARM state.
#
# Set this to the name of an existing CloudWatch alarm.
#
DEPLOYMENT_ROLLBACK_ALARM_NAME = "my-diabetes-endpoint-5xx-alarm"


# ------------------------------------------------------------
# Optional: Deployment Rollback
# ------------------------------------------------------------

ENABLE_AUTO_ROLLBACK = True

