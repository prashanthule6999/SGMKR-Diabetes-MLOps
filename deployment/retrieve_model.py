import boto3
import logging
from typing import Any
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sm_client = boto3.client("sagemaker")


def get_latest_approved_model(
    model_package_group_name: str,
) -> dict[str, Any]:
    """
    Return the latest approved Model Package from a SageMaker
    Model Package Group.

    Args:
        model_package_group_name:
            Name of the SageMaker Model Package Group.

    Returns:
        Dictionary describing the latest approved Model Package.

    Raises:
        ValueError:
            If no approved model exists.

        ClientError:
            If the SageMaker API call fails.
    """

    try:

        paginator = sm_client.get_paginator(
            "list_model_packages"
        )

        for page in paginator.paginate(
            ModelPackageGroupName=model_package_group_name,
            SortBy="CreationTime",
            SortOrder="Descending",
        ):

            for package in page["ModelPackageSummaryList"]:

                if package["ModelApprovalStatus"] == "Approved":

                    logger.info(
                        "Latest approved model found. Version: %s, ARN: %s",
                        package["ModelPackageVersion"],
                        package["ModelPackageArn"],
                    )

                    return package

        raise ValueError(
            f"No approved model found in Model Package Group "
            f"'{model_package_group_name}'."
        )

    except ClientError:
        logger.exception(
            "Failed to retrieve approved model from Model Package Group '%s'.",
            model_package_group_name,
        )
        raise
