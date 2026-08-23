import boto3
import logging

from config import (
    ENDPOINT_NAME,
    SNS_TOPIC_ARN,
    MONITOR_SCHEDULE_NAME,
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# CloudWatch Client
# --------------------------------------------------

cloudwatch = boto3.client(
    "cloudwatch"
)


# ==================================================
# DATA QUALITY ALARM
# ==================================================

def create_data_quality_alarm():

    logger.info(
        "Creating Data Quality alarm..."
    )

    cloudwatch.put_metric_alarm(

        AlarmName=
            "diabetes-data-quality-alarm",

        AlarmDescription=
            "Triggers when SageMaker Model Monitor detects data quality constraint violations.",

        Namespace=
            "aws/sagemaker/Endpoints",

        MetricName=
            "ConstraintViolations",

        Dimensions=[

            {
                "Name":
                    "MonitoringScheduleName",

                "Value":
                    MONITOR_SCHEDULE_NAME,
            },

            {
                "Name":
                    "EndpointName",

                "Value":
                    ENDPOINT_NAME,
            },
        ],

        Statistic=
            "Maximum",

        Period=
            300,

        EvaluationPeriods=
            1,

        Threshold=
            1,

        ComparisonOperator=
            "GreaterThanOrEqualToThreshold",

        TreatMissingData=
            "notBreaching",

        ActionsEnabled=
            True,

        AlarmActions=[
            SNS_TOPIC_ARN
        ],
    )

    logger.info(
        "Data Quality alarm created."
    )


# ==================================================
# DATA DRIFT ALARM
# ==================================================

def create_data_drift_alarm():

    logger.info(
        "Creating Data Drift alarm..."
    )

    cloudwatch.put_metric_alarm(

        AlarmName=
            "diabetes-data-drift-alarm",

        AlarmDescription=
            "Triggers when significant feature data drift is detected.",

        Namespace=
            "Diabetes/ModelMonitoring",

        MetricName=
            "DataDriftDetected",

        Dimensions=[

            {
                "Name":
                    "EndpointName",

                "Value":
                    ENDPOINT_NAME,
            }
        ],

        Statistic=
            "Maximum",

        Period=
            300,

        EvaluationPeriods=
            1,

        Threshold=
            1,

        ComparisonOperator=
            "GreaterThanOrEqualToThreshold",

        TreatMissingData=
            "notBreaching",

        ActionsEnabled=
            True,

        AlarmActions=[
            SNS_TOPIC_ARN
        ],
    )

    logger.info(
        "Data Drift alarm created."
    )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    create_data_quality_alarm()

    create_data_drift_alarm()

    logger.info(
        "All monitoring alarms created successfully."
    )