"""
Create a SageMaker Model resource from an approved Model Package.

Args:
    project_name:
        Project name used to tag AWS resources.

    model_package:
        Dictionary returned by retrieve_model.py
        describing the approved Model Package.

    model_name_prefix:
        Prefix used when constructing the SageMaker
        Model name.

    execution_role:
        SageMaker execution role ARN.

Returns:
    Name of the created SageMaker Model.
"""


import boto3
import logging
from typing import Any
from botocore.exceptions import ClientError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


sm_client = boto3.client("sagemaker")


def create_model(
    project_name: str,
    model_package: dict[str, Any],
    model_name_prefix: str,
    execution_role: str,
) -> str:
    """
    Create a SageMaker Model Resource from an approved Model Package. 
    The Model resource references the registered model artifacts, inference container, 
    and execution role, making it ready for deployment.

    Args:
        model_package: Dictionary returned by retrieve_model.py.
        model_name_prefix: Prefix for the SageMaker model name.
        execution_role: SageMaker execution role ARN.

    Returns:
        Name of the created SageMaker Model.
    """

    try:
        # Extract information from the Model Package
        model_package_arn = model_package["ModelPackageArn"]
        model_version = model_package["ModelPackageVersion"]

        # Create a unique SageMaker Model name
        model_name = (
            f"{model_name_prefix}-v{model_version}"
        )

        logger.info(
            "Creating SageMaker Model '%s' from Model Package version %s.",
            model_name,
            model_version,
        )

        # Create the SageMaker Model
        response = sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=execution_role,
            Containers=[
                {
                    "ModelPackageName": model_package_arn
                }
            ],
            Tags=[
                {"Key": "Project", "Value": project_name},
                {"Key": "ModelVersion", "Value": str(model_version)},
                {"Key": "Environment", "Value": "prod"}
            ],
        )

        logger.info(
            "Created SageMaker Model '%s'. ARN: %s",
            model_name,
            response["ModelArn"],
        )

        return model_name

    except ClientError:

        logger.exception(
            "Failed to create SageMaker Model '%s'.",
            model_name,
        )

        raise
