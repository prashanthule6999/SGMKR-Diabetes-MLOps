import boto3
import logging


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


sns = boto3.client("sns")


TOPIC_NAME = "diabetes-model-monitoring"


response = sns.create_topic(
    Name=TOPIC_NAME
)


topic_arn = response["TopicArn"]


logger.info(
    "SNS Topic created: %s",
    topic_arn,
)