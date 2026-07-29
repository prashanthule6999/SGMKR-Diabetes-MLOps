"""
Utility functions for inspecting the current SageMaker deployment state.

These functions are read-only. They never create, update,
or delete AWS resources.

Responsibilities:
- Check whether a SageMaker Model exists.
- Check whether a SageMaker Endpoint exists.
- Determine which model an Endpoint is serving.
- Determine whether a deployment is required.
"""

import boto3
import logging
from botocore.exceptions import ClientError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


sm_client = boto3.client("sagemaker")


def get_endpoint_model(endpoint_name: str) -> str | None:
    """
    Returns the SageMaker Model currently serving an endpoint.

    Returns:
        Model name or None if endpoint doesn't exist.
    """

    try:

        endpoint = sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )

        endpoint_config_name = endpoint["EndpointConfigName"]

        endpoint_config = sm_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )

        return endpoint_config["ProductionVariants"][0]["ModelName"]

    except ClientError as e:

        error = e.response["Error"]["Code"]

        if error == "ValidationException":
            logger.info(
                "Endpoint '%s' does not exist.",
                endpoint_name,
            )
            return None

        raise


def endpoint_already_serving_model(
    endpoint_name: str,
    desired_model_name: str,
) -> bool:
    """
    Returns True if endpoint is already serving
    the desired model.
    """

    current_model = get_endpoint_model(
        endpoint_name
    )

    if current_model is None:
        return False

    logger.info(
        "Endpoint serving model '%s'. Desired model '%s'.",
        current_model,
        desired_model_name,
    )

    return current_model == desired_model_name


def model_exists(model_name: str) -> bool:
    """
    Check if a SageMaker Model already exists.
    """

    try:
        sm_client.describe_model(
            ModelName=model_name
        )

        logger.info(
            "Model '%s' already exists.",
            model_name,
        )

        return True

    except ClientError as e:

        error = e.response["Error"]["Code"]

        if error == "ValidationException":
            return False

        raise


def endpoint_exists(endpoint_name: str) -> bool:
    """
    Check whether a SageMaker Endpoint exists.

    Args:
        endpoint_name:
            SageMaker Endpoint name.

    Returns:
        True if the endpoint exists, otherwise False.
    """

    try:

        sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )

        return True

    except ClientError as e:

        if e.response["Error"]["Code"] == "ValidationException":
            return False

        raise
