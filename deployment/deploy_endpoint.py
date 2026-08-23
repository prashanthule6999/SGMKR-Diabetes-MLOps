"""
Create or update a SageMaker Endpoint.

Responsibilities
----------------
1. Determine whether the endpoint already exists.
2. Create the endpoint if it does not exist.
3. Update the endpoint if it already exists.
4. Wait until the endpoint reaches the InService state.
5. Raise an exception if deployment fails.
"""

import boto3
import logging
from botocore.exceptions import ClientError

from deployment.deployment_utils import endpoint_exists

from config import (
    DEPLOYMENT_STRATEGY,
    CANARY_SIZE_PERCENT,
    CANARY_WAIT_INTERVAL_SECONDS,
    TERMINATION_WAIT_SECONDS,
    MAXIMUM_EXECUTION_TIMEOUT_SECONDS,
    DEPLOYMENT_ROLLBACK_ALARM_NAME,
    ENABLE_AUTO_ROLLBACK,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sm_client = boto3.client("sagemaker")


def create_or_update_endpoint(
    endpoint_name: str,
    endpoint_config_name: str,
) -> None:
    """
    Create or update a SageMaker Endpoint.

    New endpoint:
        Creates the endpoint normally.

    Existing endpoint:
        Performs a Blue/Green Canary deployment.
    """

    try:

        # ==================================================
        # Existing Endpoint
        # ==================================================

        if endpoint_exists(endpoint_name):

            logger.info(
                "Endpoint '%s' already exists.",
                endpoint_name,
            )

            deployment_config = (
                build_deployment_config()
            )

            logger.info(
                "Starting %s deployment for endpoint '%s'.",
                DEPLOYMENT_STRATEGY,
                endpoint_name,
            )

            logger.info(
                "New Endpoint Configuration: %s",
                endpoint_config_name,
            )

            logger.info(
                "Deployment configuration: %s",
                deployment_config,
            )

            sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                DeploymentConfig=deployment_config,
            )

        # ==================================================
        # New Endpoint
        # ==================================================

        else:

            logger.info(
                "Endpoint '%s' does not exist.",
                endpoint_name,
            )

            logger.info(
                "Creating endpoint using Endpoint "
                "Configuration '%s'.",
                endpoint_config_name,
            )

            sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )

        # ==================================================
        # Wait for deployment
        # ==================================================

        logger.info(
            "Waiting for endpoint '%s' to reach "
            "InService status...",
            endpoint_name,
        )

        waiter = sm_client.get_waiter(
            "endpoint_in_service"
        )

        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={
                "Delay": 30,
                "MaxAttempts": 60,
            },
        )

        logger.info(
            "Endpoint '%s' is now InService.",
            endpoint_name,
        )

    except ClientError:

        logger.exception(
            "Failed to deploy endpoint '%s'.",
            endpoint_name,
        )

        raise

    def build_deployment_config() -> dict:
        """
        Build SageMaker Blue/Green deployment configuration.

        Current strategy:
            CANARY

        The configuration is intentionally kept separate from
        the endpoint update logic so deployment.py remains clean.
        """

        if DEPLOYMENT_STRATEGY != "CANARY":
            raise ValueError(
                f"Unsupported deployment strategy: "
                f"{DEPLOYMENT_STRATEGY}"
            )

        deployment_config = {
            "BlueGreenUpdatePolicy": {
                "TrafficRoutingConfiguration": {
                    "Type": "CANARY",

                    "CanarySize": {
                        "Type": "CAPACITY_PERCENT",
                        "Value": CANARY_SIZE_PERCENT,
                    },

                    "WaitIntervalInSeconds": (
                        CANARY_WAIT_INTERVAL_SECONDS
                    ),
                },

                "TerminationWaitInSeconds": (
                    TERMINATION_WAIT_SECONDS
                ),

                "MaximumExecutionTimeoutInSeconds": (
                    MAXIMUM_EXECUTION_TIMEOUT_SECONDS
                ),
            }
        }

        # --------------------------------------------------
        # Automatic rollback
        # --------------------------------------------------

        if ENABLE_AUTO_ROLLBACK:

            if not DEPLOYMENT_ROLLBACK_ALARM_NAME:
                raise ValueError(
                    "ENABLE_AUTO_ROLLBACK is True but "
                    "DEPLOYMENT_ROLLBACK_ALARM_NAME is empty."
                )

            deployment_config[
                "AutoRollbackConfiguration"
            ] = {
                "Alarms": [
                    {
                        "AlarmName": (
                            DEPLOYMENT_ROLLBACK_ALARM_NAME
                        )
                    }
                ]
            }

        return deployment_config
