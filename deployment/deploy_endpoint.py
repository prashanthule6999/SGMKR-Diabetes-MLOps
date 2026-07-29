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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sm_client = boto3.client("sagemaker")


def create_or_update_endpoint(
    endpoint_name: str,
    endpoint_config_name: str,
) -> None:
    """
    Create or update a SageMaker Endpoint.

    If the endpoint already exists it is updated to use the
    supplied Endpoint Configuration. Otherwise a new endpoint
    is created.

    Args:
        endpoint_name:
            SageMaker Endpoint name.

        endpoint_config_name:
            Endpoint Configuration to deploy.
    """

    try:

        if endpoint_exists(endpoint_name):

            logger.info(
                "Updating endpoint '%s' using Endpoint Configuration '%s'.",
                endpoint_name,
                endpoint_config_name,
            )

            sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )

        else:

            logger.info(
                "Creating endpoint '%s' using Endpoint Configuration '%s'.",
                endpoint_name,
                endpoint_config_name,
            )

            sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )

        logger.info(
            "Waiting for endpoint '%s' to reach InService status...",
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
