import sagemaker

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep

from sagemaker.workflow.functions import Join
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo

from sagemaker.model_metrics import ModelMetrics
from sagemaker.model_metrics import MetricsSource
from sagemaker.workflow.step_collections import RegisterModel

from sagemaker.sklearn.model import SKLearnModel
from config import EXECUTION_ROLE_ARN

ROLE = EXECUTION_ROLE_ARN
from config import BUCKET

raw_data_s3_uri = f"s3://{BUCKET}/raw"

pipeline_session = PipelineSession()

# defining the execution environment for processing job
processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipeline_session
)

# Execute this script using above processor.
preprocessing_step = ProcessingStep(
    name="DataPreprocessing",
    processor=processor,
    code="processing/preprocessing.py",

    inputs=[
        ProcessingInput(
            source=raw_data_s3_uri,  # S3 location
            destination="/opt/ml/processing/input",  # Container location
        )
    ],

    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train"  # Container location
            # Pipeline managed output is used which is
            # handling output storage location so
            # no destination required
        ),

        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/validation"  # Container location
            # Pipeline managed output is used which is
            # handling output storage location so
            # no destination required
        ),

        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test"  # Container
            # Pipeline managed output is used which is
            # handling output storage location so
            # no destination required
        )
    ],

    job_arguments=[
        "--input-data", "/opt/ml/processing/input",
        "--train-output", "/opt/ml/processing/train",
        "--validation-output", "/opt/ml/processing/validation",
        "--test-output", "/opt/ml/processing/test",

        "--input-file", "diabetes.csv",
        "--test-size", "0.2",
        "--val-size", "0.25",
        "--random-state", "42",
        "--target-column", "Outcome",
        "--columns", "Glucose,BloodPressure,SkinThickness,Insulin,BMI"
    ]
)

# defining the execution environment for training job
estimator = SKLearn(
    entry_point="training/train.py",
    framework_version="1.2-1",
    role=ROLE,
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=pipeline_session,
    hyperparameters={
        "target-column": "Outcome",
        "penalty": "l2",
        "solver": "liblinear",
        "C": 0.1,
        "class-weight": "None",
        "random-state": 42
    }
)

# Execute this script using above estimator.
training_step = TrainingStep(
    name="ModelTraining",
    estimator=estimator,

    # Channels - train & validation
    inputs={
        "train": TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "train"].S3Output.S3Uri
            # don't care where SageMaker stored previous step o/p
            # the pipeline connects the steps automatically.
        ),
        "validation": TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "validation"].S3Output.S3Uri
        )
    }
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

evaluation_step = ProcessingStep(
    name="ModelEvaluation",
    processor=processor,
    code="processing/evaluate.py",
    property_files=[evaluation_report],

    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],

    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ],

    job_arguments=[
        "--input-model-artifacts", "/opt/ml/processing/model",
        "--input-test-data", "/opt/ml/processing/test",
        "--evaluation-output", "/opt/ml/processing/evaluation",
        "--target-column", "Outcome"
    ]
)

# SageMaker needs to know where that file is to uploads your evaluation.json to the Model Registry
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="/",
            values=[
                evaluation_step.properties
                .ProcessingOutputConfig
                .Outputs["evaluation"]
                .S3Output.S3Uri,
                "evaluation.json",
            ],
        ),
        content_type="application/json",
    )
)

model = SKLearnModel(
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=ROLE,
    entry_point="inference.py",
    source_dir="inference",
    framework_version="1.2-1",
    py_version="py3",
)

register_step = RegisterModel(

    name="RegisterDiabetesModel",
    model=model,

    content_types=[
        "application/json"
    ],

    response_types=[
        "application/json"
    ],

    inference_instances=[
        "ml.t2.medium"
    ],

    transform_instances=[
        "ml.m5.large"
    ],

    model_package_group_name="DiabetesPredictionModel",
    approval_status="PendingManualApproval",
    model_metrics=model_metrics,
)

condition_step = ConditionStep(
    name="ModelValidation",

    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.accuracy.value",
            ),
            right=0.70,
        ),
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.precision.value",
            ),
            right=0.55,
        ),
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.recall.value",
            ),
            right=0.80,
        ),
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.f1.value",
            ),
            right=0.65,
        ),
    ],

    if_steps=[register_step],

    else_steps=[
        # Pipeline stops if any condition fails
    ],
)

pipeline = Pipeline(
    name="DiabetesPipeline",
    steps=[preprocessing_step,
           training_step,
           evaluation_step,
           condition_step],
    sagemaker_session=pipeline_session,
)
