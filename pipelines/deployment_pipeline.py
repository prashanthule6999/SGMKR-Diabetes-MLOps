# Defines the deployment workflow
import logging
from datetime import datetime

from deployment.retrieve_model import get_latest_approved_model

from deployment.deployment_utils import (
    model_exists,
    endpoint_already_serving_model,
)

from deployment.create_model import create_model
from deployment.create_endpoint_config import create_endpoint_config
from deployment.deploy_endpoint import create_or_update_endpoint

from config import (
    MODEL_PACKAGE_GROUP_NAME,
    MODEL_NAME_PREFIX,
    ENDPOINT_NAME,
    PROJECT_NAME,
    EXECUTION_ROLE_ARN,
    INSTANCE_TYPE,
    INITIAL_INSTANCE_COUNT,
    PRODUCTION_MONITORING_DATA_S3_URI,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Deployment Workflow

    1. Retrieve latest approved Model Package.
    2. Build version-specific SageMaker Model name.
    3. Check whether endpoint already serves that model.
    4. Stop if the desired model is already deployed.
    5. Create or reuse the SageMaker Model.
    6. Create a new immutable Endpoint Configuration.
    7. Create or update the Endpoint.
    8. Existing endpoint → Blue/Green Canary deployment.
    9. New endpoint → normal creation.
    """

    try:

        logger.info("Starting deployment pipeline...")

        # --------------------------------------------------
        # Retrieve latest approved model package
        # --------------------------------------------------
        approved_model = get_latest_approved_model(
            MODEL_PACKAGE_GROUP_NAME
        )

        model_version = approved_model["ModelPackageVersion"]

        desired_model_name = (
            f"{MODEL_NAME_PREFIX}-v{model_version}"
        )

        logger.info(
            "Desired SageMaker Model: %s",
            desired_model_name,
        )

        logger.info(
            "Latest approved model version: %s",
            model_version,
        )

        # --------------------------------------------------
        # Is endpoint already serving this model?
        # --------------------------------------------------
        if endpoint_already_serving_model(
            endpoint_name=ENDPOINT_NAME,
            desired_model_name=desired_model_name,
        ):

            logger.info(
                "Endpoint already serving the latest approved model. Nothing to deploy."
            )

            return

        # --------------------------------------------------
        # Create or reuse SageMaker Model
        # --------------------------------------------------
        if model_exists(desired_model_name):

            logger.info(
                "Reusing existing SageMaker Model: %s",
                desired_model_name,
            )

            model_name = desired_model_name

        else:

            logger.info(
                "Creating SageMaker Model..."
            )

            model_name = create_model(
                project_name=PROJECT_NAME,
                model_package=approved_model,
                model_name_prefix=MODEL_NAME_PREFIX,
                execution_role=EXECUTION_ROLE_ARN,
            )

            logger.info(
                "Using SageMaker Model: %s",
                model_name,
            )

        # --------------------------------------------------
        # Generate unique Endpoint Configuration name
        # --------------------------------------------------
        endpoint_config_name = (
            f"{ENDPOINT_NAME}-config-v{model_version}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        logger.info(
            "Endpoint configuration: %s",
            endpoint_config_name,
        )

        # --------------------------------------------------
        # Create Endpoint Configuration
        # --------------------------------------------------
        create_endpoint_config(
            project_name=PROJECT_NAME,
            endpoint_config_name=endpoint_config_name,
            model_name=model_name,
            instance_type=INSTANCE_TYPE,
            initial_instance_count=INITIAL_INSTANCE_COUNT,
            data_capture_s3_uri=PRODUCTION_MONITORING_DATA_S3_URI
        )

        # --------------------------------------------------
        # Create or Update Endpoint
        # --------------------------------------------------
        create_or_update_endpoint(
            endpoint_name=ENDPOINT_NAME,
            endpoint_config_name=endpoint_config_name,
        )

        logger.info(
            "Deployment completed successfully.\n"
            "Model Version: %s\n"
            "Model: %s\n"
            "Endpoint Config: %s\n"
            "Endpoint: %s",
            model_version,
            model_name,
            endpoint_config_name,
            ENDPOINT_NAME,
        )

    except Exception:

        logger.exception(
            "Deployment failed."
        )

        raise
