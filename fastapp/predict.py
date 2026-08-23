# ------------------------------------------------------------------------------
# FastAPI will simply send the request to the SageMaker Endpoint
# FastAPI validates the request (Pydantic),
# and invokes the SageMaker endpoint, and returns the response
# ------------------------------------------------------------------------------
import json
import logging
import boto3
import uuid

from botocore.config import Config
from botocore.exceptions import ClientError

from config import (
    ENDPOINT_NAME,
    AWS_REGION,
)

logger = logging.getLogger(__name__)

config = Config(
    retries={
        "max_attempts": 3,
        "mode": "standard",
    }
)

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=AWS_REGION,
    config=config,
)

sm_client = boto3.client(
    "sagemaker",
    region_name=AWS_REGION,
    config=config,
)

def predict_output(user_input: dict) -> dict:

    try:

        # --------------------------------------------------
        # Generate unique ID for this inference
        # --------------------------------------------------

        inference_id = str(uuid.uuid4())


        # --------------------------------------------------
        # Invoke SageMaker Endpoint
        # --------------------------------------------------

        response = runtime.invoke_endpoint(

            EndpointName=ENDPOINT_NAME,

            ContentType="application/json",

            Body=json.dumps(user_input),

            InferenceId=inference_id,
        )


        status_code = (
            response["ResponseMetadata"]["HTTPStatusCode"]
        )


        if status_code != 200:

            raise RuntimeError(
                f"SageMaker invocation failed "
                f"with HTTP {status_code}"
            )


        result = json.loads(
            response["Body"].read().decode("utf-8")
        )


        logger.info(
            "Prediction successful. "
            "InferenceId=%s",
            inference_id,
        )


        # --------------------------------------------------
        # Return prediction + inference ID
        # --------------------------------------------------

        return {

            "prediction": result.get(
                "prediction"
            ),

            "inference_id": inference_id,

        }


    except ClientError:

        logger.exception(
            "AWS SageMaker invocation failed."
        )

        raise


    except Exception:

        logger.exception(
            "Prediction failed."
        )

        raise
    
def check_endpoint(endpoint_name: str) -> str:

    try:

        response = sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )

        status = response["EndpointStatus"]

        if status != "InService":
            raise RuntimeError(
                f"Endpoint status: {status}"
            )

        logger.info(
            "Endpoint '%s' is %s.",
            endpoint_name,
            status,
        )

        return status

    except ClientError:

        logger.exception(
            "Failed to describe endpoint '%s'.",
            endpoint_name,
        )

        raise

    except Exception:

        logger.exception(
            "Endpoint health check failed."
        )

        raise
