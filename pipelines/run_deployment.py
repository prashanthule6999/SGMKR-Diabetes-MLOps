# Executes the deployment workflow
"""
Entry point for deploying the latest approved SageMaker model.

This script is intended to be executed by CI/CD (GitHub Actions)
or manually from the command line.

Workflow
--------
1. Import the deployment pipeline.
2. Execute deployment.
3. Exit with success/failure status.
"""

import logging
import sys

from deployment_pipeline import main


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def run():

    logger.info("Starting deployment workflow...")

    try:

        main()

        logger.info(
            "Deployment workflow completed successfully."
        )

    except Exception:

        logger.exception(
            "Deployment workflow failed."
        )

        sys.exit(1)


if __name__ == "__main__":
    run()