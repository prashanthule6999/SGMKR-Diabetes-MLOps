import logging
import sagemaker
from training_pipeline import pipeline
from config import EXECUTION_ROLE_ARN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline.upsert(
    role_arn=EXECUTION_ROLE_ARN
)


def main():

    logger.info("Creating/Updating SageMaker Pipeline...")

    pipeline.upsert(role_arn=EXECUTION_ROLE_ARN)

    logger.info("Starting Pipeline Execution...")

    execution = pipeline.start()

    logger.info(
        "Pipeline Execution ARN: %s",
        execution.arn,
    )


if __name__ == "__main__":
    main()