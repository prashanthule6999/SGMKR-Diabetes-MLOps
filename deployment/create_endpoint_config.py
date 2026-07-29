# This file creates a SageMaker Endpoint Configuration.
#
# An Endpoint Configuration defines how a SageMaker Model
# should be deployed, including:
#
# - Which SageMaker Model to deploy
# - Which instance type to use
# - How many instances to launch
# - Traffic routing across Production Variants
#
# Endpoint Configurations are immutable.
# A new Endpoint Configuration must be created whenever
# the model version or deployment settings change.

"""
Creates a new immutable SageMaker Endpoint Configuration
that references a SageMaker Model and specifies the
deployment settings.

A new Endpoint Configuration must be created whenever
deployment settings or model version change.
"""

import boto3
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sm_client = boto3.client("sagemaker")


def create_endpoint_config(
    project_name: str,
    endpoint_config_name: str,
    model_name: str,
    instance_type: str = "ml.t2.medium",  # Which machine should host my model?
    initial_instance_count: int = 1,  # How many copies of your endpoint should run?
) -> None:
    """
    Create a SageMaker Endpoint Configuration.

    Args:
        project_name:
            Project name used for tagging AWS resources.
        endpoint_config_name:
            Unique Endpoint Configuration name.
        model_name:
            SageMaker Model name.
        instance_type:
            Instance type used for inference.
        initial_instance_count:
            Number of inference instances.

    """

    try:

        logger.info(
            "Creating Endpoint Configuration: %s",
            endpoint_config_name,
        )

        sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": initial_instance_count,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
            Tags=[
                {
                    "Key": "Project",
                    "Value": project_name,
                }
            ],
        )

        logger.info(
            "Created Endpoint Configuration %s",
            endpoint_config_name,
        )

    except ClientError:
        logger.exception(
            "Failed to create Endpoint Configuration %s",
            endpoint_config_name,
        )
        raise
