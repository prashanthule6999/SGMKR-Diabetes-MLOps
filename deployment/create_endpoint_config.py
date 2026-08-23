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
Creates a new immutable SageMaker Endpoint Configuration.

The Endpoint Configuration defines:
- Which SageMaker Model to deploy
- Which instance type to use
- Number of inference instances
- Traffic routing
- Data Capture configuration
"""

import boto3
import logging

from config import *
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
    data_capture_s3_uri: str = None
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
        data_capture_s3_uri:
            S3 location where SageMaker stores
            captured inference requests and responses.
    """

    try:

        logger.info(
            "Creating Endpoint Configuration: %s",
            endpoint_config_name,
        )

        # --------------------------------------------------
        # Data Capture Configuration
        # --------------------------------------------------

        data_capture_config = None

        if data_capture_s3_uri:

            data_capture_config = {
                "EnableCapture": True,

                "InitialSamplingPercentage": DATA_CAPTURE_SAMPLING_PERCENTAGE,

                "DestinationS3Uri": data_capture_s3_uri,

                "CaptureOptions": [
                    {
                        "CaptureMode": "Input"
                    },
                    {
                        "CaptureMode": "Output"
                    }
                ],

                "CaptureContentTypeHeader": {
                    "CsvContentTypes": [
                        "text/csv"
                    ],
                    "JsonContentTypes": [
                        "application/json"
                    ]
                }
            }

            logger.info(
                "Data Capture enabled. S3 destination: %s",
                data_capture_s3_uri,
            )

        else:

            logger.info(
                "Data Capture disabled."
            )

        # --------------------------------------------------
        # Create Endpoint Configuration
        # --------------------------------------------------

        request = {
            "EndpointConfigName": endpoint_config_name,

            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": initial_instance_count,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],

            "Tags": [
                {
                    "Key": "Project",
                    "Value": project_name,
                }
            ],
        }

        # Add DataCaptureConfig only when enabled
        if data_capture_config:

            request[
                "DataCaptureConfig"
            ] = data_capture_config

        sm_client.create_endpoint_config(
            **request
        )

        logger.info(
            "Created Endpoint Configuration: %s",
            endpoint_config_name,
        )

    except ClientError:

        logger.exception(
            "Failed to create Endpoint Configuration: %s",
            endpoint_config_name,
        )

        raise
