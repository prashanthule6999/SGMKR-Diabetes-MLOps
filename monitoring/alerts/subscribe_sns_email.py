import boto3


sns = boto3.client("sns")


SNS_TOPIC_ARN = (
    "YOUR_SNS_TOPIC_ARN"
)

EMAIL = (
    "your-email@example.com"
)


response = sns.subscribe(

    TopicArn=SNS_TOPIC_ARN,

    Protocol="email",

    Endpoint=EMAIL,
)


print(
    "Subscription created."
)

print(
    "Check your email and confirm the subscription."
)