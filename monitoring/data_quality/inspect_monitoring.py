import logging
import sagemaker

from config import *
from sagemaker.model_monitor import DefaultModelMonitor


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
# Inspect Latest Monitoring Execution
# --------------------------------------------------

def inspect_latest_monitoring_execution():

    logger.info(
        "Fetching monitoring executions..."
    )

    executions = monitor.list_executions()

    if not executions:

        logger.warning(
            "No monitoring executions found."
        )

        return {
            "status": "NO_EXECUTION",
            "violation_count": 0,
            "violations": [],
        }


    # --------------------------------------------------
    # Latest execution
    # --------------------------------------------------

    latest_execution = executions[0]

    details = latest_execution.describe()


    processing_status = details.get(
        "ProcessingJobStatus"
    )

    monitoring_status = details.get(
        "MonitoringExecutionStatus"
    )


    logger.info(
        "Processing status: %s",
        processing_status,
    )

    logger.info(
        "Monitoring status: %s",
        monitoring_status,
    )


    # --------------------------------------------------
    # Check execution status
    # --------------------------------------------------

    if processing_status not in [
        "Completed",
        "CompletedWithViolations",
    ]:

        logger.warning(
            "Monitoring execution is not completed."
        )

        return {
            "status": processing_status,
            "violation_count": 0,
            "violations": [],
        }


    # --------------------------------------------------
    # Get constraint violations
    # --------------------------------------------------

    violations = (
        latest_execution
        .constraint_violations()
    )

    violation_data = violations.body_dict

    violation_list = violation_data.get(
        "violations",
        []
    )


    # --------------------------------------------------
    # Return result
    # --------------------------------------------------

    return {

        "status": processing_status,

        "monitoring_status": monitoring_status,

        "violation_count": len(
            violation_list
        ),

        "violations": violation_list,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    result = (
        inspect_latest_monitoring_execution()
    )

    print("\n")
    print("=" * 60)
    print("DATA QUALITY MONITORING RESULT")
    print("=" * 60)

    print(
        "Status:",
        result["status"]
    )

    print(
        "Violation count:",
        result["violation_count"]
    )

    print("=" * 60)