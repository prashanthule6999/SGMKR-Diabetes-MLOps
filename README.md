# 🩺 Diabetes Prediction MLOps Platform

<p align="center">
  <strong>End-to-End Machine Learning Operations Platform on AWS SageMaker</strong>
</p>

<p align="center">
  Train → Evaluate → Register → Approve → Deploy → Serve → Monitor → Alert
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-SageMaker-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![SageMaker](https://img.shields.io/badge/Amazon-SageMaker-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

</p>

---

## 📌 Overview

This project implements an end-to-end **production-oriented MLOps platform** for a Diabetes Prediction machine learning model using **Amazon SageMaker**.

The project demonstrates how a machine learning model can move from development to production through a controlled and repeatable lifecycle:

```text
Raw Data
   ↓
Data Processing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Quality Validation
   ↓
Model Registry
   ↓
Model Approval
   ↓
Deployment
   ↓
Real-Time Inference
   ↓
Production Data Capture
   ↓
Data Quality Monitoring
   ↓
Data Drift Monitoring
   ↓
Model Quality Monitoring
   ↓
CloudWatch
   ↓
SNS Alerting
````

The goal of this project is not simply to train a machine learning model.

The goal is to demonstrate how a machine learning model can be:

* Reproducibly trained
* Automatically evaluated
* Quality-gated
* Versioned
* Approved
* Safely deployed
* Served through an API
* Monitored in production
* Observed through metrics
* Alerted when production issues occur

---

# 🎯 Project Goals

The platform was designed around the following MLOps principles:

### 1. Separation of Training and Deployment

Training and deployment are handled independently.

```text
Training Pipeline
       ↓
Model Registry
       ↓
Approved Model
       ↓
Deployment Pipeline
```

This prevents every training execution from automatically changing production.

---

### 2. Model Governance

Models are registered and versioned using SageMaker Model Registry.

Only an **Approved Model Package** is considered for deployment.

---

### 3. Safe Deployment

The deployment pipeline checks the current production state before making changes.

It avoids unnecessary deployment when the desired model is already serving traffic.

---

### 4. Production Monitoring

The deployed model is monitored from multiple perspectives:

```text
Data Quality
     +
Data Drift
     +
Model Quality
```

---

### 5. Application Observability

The FastAPI layer exposes Prometheus metrics for application-level observability.

---

# 🏗️ High-Level Architecture

```text
                                      ┌───────────────────┐
                                      │      GitHub       │
                                      │   Source Code     │
                                      └─────────┬─────────┘
                                                │
                              ┌─────────────────┴─────────────────┐
                              │                                   │
                              ▼                                   ▼
                       ┌────────────┐                     ┌────────────┐
                       │  train.yml │                     │ deploy.yml │
                       │  CI/CD     │                     │  CI/CD     │
                       └─────┬──────┘                     └─────┬──────┘
                             │                                  │
                             ▼                                  │
                  ┌──────────────────────┐                     │
                  │ SageMaker Training   │                     │
                  │ Pipeline              │                     │
                  └──────────┬───────────┘                     │
                             │                                  │
             ┌───────────────┼────────────────┐                 │
             │               │                │                 │
             ▼               ▼                ▼                 │
       ┌──────────┐    ┌──────────┐    ┌────────────┐          │
       │Processing│    │ Training │    │ Evaluation │          │
       │   Job    │    │   Job    │    │    Job     │          │
       └────┬─────┘    └────┬─────┘    └─────┬──────┘          │
            │               │                │                 │
            └───────────────┼────────────────┘                 │
                            ▼                                  │
                    ┌──────────────┐                           │
                    │ Condition    │                           │
                    │ Step         │                           │
                    └──────┬───────┘                           │
                           │                                   │
                    ┌──────┴──────┐                            │
                    │             │                            │
                   PASS          FAIL                          │
                    │             │                            │
                    ▼             ▼                            │
             ┌────────────┐     STOP                           │
             │   Model    │                                    │
             │  Registry  │                                    │
             └──────┬─────┘                                    │
                    │                                          │
                    ▼                                          │
             Approved Model                                    │
                    │                                          │
                    └──────────────────────────────────────────┘
                                                               │
                                                               ▼
                                                    ┌─────────────────────┐
                                                    │ Deployment Workflow │
                                                    └──────────┬──────────┘
                                                               │
                                                               ▼
                                                    ┌─────────────────────┐
                                                    │ Deployment Guard     │
                                                    │                     │
                                                    │ Model Exists?       │
                                                    │ Endpoint Exists?    │
                                                    │ Same Model Serving? │
                                                    └──────────┬──────────┘
                                                               │
                                                               ▼
                                                    ┌─────────────────────┐
                                                    │  SageMaker Model    │
                                                    └──────────┬──────────┘
                                                               │
                                                               ▼
                                                    ┌─────────────────────┐
                                                    │ Endpoint Config     │
                                                    │                     │
                                                    │ Model               │
                                                    │ Instance Type       │
                                                    │ Instance Count      │
                                                    │ Data Capture        │
                                                    └──────────┬──────────┘
                                                               │
                                                               ▼
                                                    ┌─────────────────────┐
                                                    │ SageMaker Endpoint  │
                                                    │                     │
                                                    │ Real-Time Inference │
                                                    └──────────┬──────────┘
                                                               │
                                              ┌────────────────┴──────────────┐
                                              │                               │
                                              ▼                               ▼
                                       ┌──────────────┐               ┌──────────────┐
                                       │   FastAPI    │               │ Data Capture │
                                       │   REST API   │               │      S3      │
                                       └──────┬───────┘               └──────┬───────┘
                                              │                              │
                                              ▼                              ▼
                                       ┌──────────────┐               ┌──────────────┐
                                       │  Prometheus  │               │ SageMaker    │
                                       │   Metrics    │               │ Model Monitor│
                                       └──────────────┘               └──────┬───────┘
                                                                                │
                                                        ┌───────────────────────┼───────────────────────┐
                                                        │                       │                       │
                                                        ▼                       ▼                       ▼
                                                 Data Quality             Data Drift            Model Quality
                                                        │                       │                       │
                                                        └───────────────────────┼───────────────────────┘
                                                                                │
                                                                                ▼
                                                                        ┌──────────────┐
                                                                        │ CloudWatch   │
                                                                        └──────┬───────┘
                                                                               │
                                                                               ▼
                                                                        ┌──────────────┐
                                                                        │ CloudWatch   │
                                                                        │ Alarm        │
                                                                        └──────┬───────┘
                                                                               │
                                                                               ▼
                                                                        ┌──────────────┐
                                                                        │     SNS      │
                                                                        │ Notification │
                                                                        └──────────────┘
```

---

# 🔄 Complete MLOps Lifecycle

```text
                         ┌──────────────┐
                         │   Raw Data   │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │ Preprocessing│
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   Training   │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Evaluation  │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │  Validation  │
                         └──────┬───────┘
                                │
                         ┌──────┴──────┐
                         │             │
                       PASS           FAIL
                         │             │
                         ▼             ▼
                  ┌────────────┐      STOP
                  │   Model    │
                  │  Registry  │
                  └─────┬──────┘
                        │
                        ▼
                  ┌────────────┐
                  │  Approval  │
                  └─────┬──────┘
                        │
                        ▼
                  ┌────────────┐
                  │ Deployment │
                  └─────┬──────┘
                        │
                        ▼
                  ┌────────────┐
                  │   Serving  │
                  └─────┬──────┘
                        │
                        ▼
                  ┌────────────┐
                  │ Monitoring │
                  └────────────┘
```

---

# 🧪 Training Pipeline

The training lifecycle is implemented using **Amazon SageMaker Pipelines**.

The training pipeline is responsible for:

1. Processing data
2. Training the model
3. Evaluating the model
4. Validating performance
5. Registering the model when quality requirements are satisfied

```text
Raw Data
   ↓
Processing Job
   ↓
Train / Validation / Test
   ↓
Training Job
   ↓
Evaluation Job
   ↓
Condition Step
   ↓
Model Registry
```

---

# 1️⃣ Data Processing

A SageMaker Processing Job is responsible for preparing the dataset.

The processing stage:

* Reads data from Amazon S3
* Cleans the dataset
* Handles missing values
* Performs preprocessing
* Splits the dataset
* Produces training data
* Produces validation data
* Produces test data
* Stores outputs in Amazon S3

Conceptually:

```text
                   Raw Dataset
                       │
                       ▼
              SageMaker Processing
                       │
              ┌────────┼────────┐
              │        │        │
              ▼        ▼        ▼
            Train   Validation  Test
```

---

# 2️⃣ Model Training

The current training implementation uses:

```text
Scikit-Learn
Logistic Regression
```

The SageMaker Training Job:

1. Receives processed training data.
2. Loads the dataset.
3. Trains the machine learning model.
4. Generates the model artifact.
5. Stores the artifact in Amazon S3.

The resulting artifact becomes part of the model version lifecycle.

---

# 3️⃣ Model Evaluation

A separate SageMaker Processing Job evaluates the trained model.

The evaluation stage calculates:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Example evaluation output:

```json
{
  "accuracy": 0.70,
  "precision": 0.55,
  "recall": 0.80,
  "f1": 0.65,
  "roc_auc": 0.81
}
```

The evaluation output is consumed by the pipeline's validation step.

---

# 4️⃣ Model Validation

The pipeline uses a **SageMaker Condition Step** as a model quality gate.

Current validation thresholds:

```text
Accuracy  >= 0.70
Precision >= 0.55
Recall    >= 0.80
F1        >= 0.65
```

Flow:

```text
                    Evaluation
                         │
                         ▼
                  Condition Step
                         │
                    ┌────┴────┐
                    │         │
                   PASS      FAIL
                    │         │
                    ▼         ▼
             Register Model   STOP
```

A model that fails the quality gate is not registered for production deployment.

---

# 5️⃣ Model Registry

Approved model versions are maintained using **Amazon SageMaker Model Registry**.

Example:

```text
DiabetesPredictionModel
│
├── Version 1
├── Version 2
├── Version 3
└── Version 4  ← Approved
```

The Model Registry provides:

* Model versioning
* Approval status
* Model governance
* Model lineage
* Controlled promotion
* Deployment source of truth

---

# 📦 Model Package vs SageMaker Model

A key part of the architecture is understanding the difference between a **Model Package** and a **SageMaker Model**.

```text
Model Registry
      │
      ▼
Model Package Version
      │
      │ Approved
      ▼
SageMaker Model
      │
      ▼
Endpoint Configuration
      │
      ▼
SageMaker Endpoint
```

### Model Package

Represents the registered, versioned model candidate.

### SageMaker Model

Represents the deployable SageMaker resource.

### Endpoint Configuration

Defines how that model should be hosted.

### Endpoint

Provides the actual real-time inference service.

This separation creates a clear boundary between:

```text
Model Governance
        ↓
Production Deployment
```

---

# 🚀 Deployment Pipeline

Training and deployment are deliberately separated.

```text
┌──────────────────────┐
│ Training Pipeline    │
└──────────┬───────────┘
           │
           ▼
    Model Registry
           │
           ▼
    Approved Model
           │
           ▼
┌──────────────────────┐
│ Deployment Pipeline  │
└──────────────────────┘
```

The deployment workflow performs the following:

```text
Retrieve Latest Approved Model
              ↓
Determine Model Version
              ↓
Check Existing Deployment
              ↓
Create / Reuse SageMaker Model
              ↓
Create Endpoint Configuration
              ↓
Create / Update Endpoint
              ↓
Wait for InService
```

---

# 🔍 Deployment Guard

The deployment process is state-aware.

Before deploying, it checks:

* Does the endpoint exist?
* Which model is currently serving?
* Does the desired SageMaker Model already exist?
* Is the endpoint already serving the desired model?

Conceptually:

```text
                 Latest Approved Model
                         │
                         ▼
                 Deployment Guard
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    Model Exists?   Endpoint Exists?  Same Model?
```

If the endpoint is already serving the desired model:

```text
Deployment skipped
```

Otherwise:

```text
Create / Reuse Model
        ↓
Create Endpoint Config
        ↓
Update Endpoint
```

This prevents unnecessary production deployments.

---

# ♻️ SageMaker Model Reuse

The deployment pipeline checks whether the desired SageMaker Model already exists.

```text
                 Desired Model
                      │
                      ▼
                Model Exists?
                  /       \
                YES        NO
                 │          │
                 ▼          ▼
               Reuse      Create
                 │          │
                 └────┬─────┘
                      ▼
               SageMaker Model
```

This avoids creating duplicate SageMaker Model resources for the same model version.

---

# ⚙️ Endpoint Configuration

SageMaker Endpoint Configurations are immutable.

Therefore, every deployment creates a new Endpoint Configuration.

The Endpoint Configuration defines:

* SageMaker Model
* Instance Type
* Instance Count
* Production Variant
* Traffic Weight
* Data Capture Configuration

Example:

```text
diabetes-endpoint-config-v4-20260823-120000
```

Current production configuration uses:

```text
Variant Name:
AllTraffic

Initial Variant Weight:
1.0
```

This represents the current standard single-variant deployment.

---

# 🟢 Production Endpoint

The SageMaker Endpoint provides real-time inference.

```text
Client
   │
   ▼
FastAPI
   │
   ▼
SageMaker Endpoint
   │
   ▼
SageMaker Model
   │
   ▼
Prediction
```

The deployment workflow waits for the endpoint to reach:

```text
InService
```

before considering deployment successful.

---

# 🔵🟢 Deployment Strategy Foundation

The deployment architecture uses SageMaker Production Variants.

Production Variants provide the foundation required for:

* Blue/Green deployment
* Canary deployment
* Weighted traffic shifting

The current implementation uses a single production variant:

```text
Production Endpoint
        │
        ▼
    AllTraffic
        │
        ▼
      Model vN
```

The architecture can be extended to multiple variants:

```text
Production Endpoint
        │
        ├────────────────┐
        │                │
        ▼                ▼
      BLUE             GREEN
    Model v1          Model v2
      90%               10%
```

A future controlled rollout could progressively shift traffic:

```text
Stage 1

BLUE  → 90%
GREEN → 10%

        ↓

Stage 2

BLUE  → 50%
GREEN → 50%

        ↓

Stage 3

BLUE  → 0%
GREEN → 100%
```

### Current Implementation

The current implementation focuses on:

* Approved model deployment
* Model version awareness
* Endpoint state validation
* Model reuse
* Immutable Endpoint Configurations
* Data capture
* Safe endpoint updates
* Production monitoring

Automated traffic shifting and automated rollback are not currently part of the deployment workflow.

---

# 🌐 FastAPI Inference Layer

FastAPI provides the application-facing REST API.

```text
Client
  │
  ▼
FastAPI
  │
  ▼
Pydantic Validation
  │
  ▼
Prediction Service
  │
  ▼
SageMaker Endpoint
  │
  ▼
Model Prediction
  │
  ▼
FastAPI Response
  │
  ▼
Client
```

FastAPI acts as the application/API layer rather than directly embedding the trained model inside the web application.

---

# ❤️ Health Endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "OK",
  "endpoint": "diabetes-endpoint",
  "endpoint_status": "InService"
}
```

This endpoint can be used to verify application and SageMaker endpoint availability.

---

# 🔮 Prediction Endpoint

```http
POST /predict
```

### Request

```json
{
  "pregnancies": 6,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 1,
  "bmi": 33.6,
  "diabetes_pedigree_function": 0.627,
  "age": 50
}
```

### Response

```json
{
  "prediction": 1,
  "confidence": 0.84,
  "class_probabilities": {
    "No Diabetes": 0.16,
    "Diabetes": 0.84
  }
}
```

---

# 🧾 Request Validation

Pydantic is used to validate incoming API requests.

```text
Client Request
      │
      ▼
Pydantic Schema
      │
   ┌──┴──┐
   │     │
 Valid  Invalid
   │     │
   ▼     ▼
Predict  Error
```

This prevents malformed requests from being passed to the inference endpoint.

---

# 📊 Application Metrics

The FastAPI application exposes Prometheus metrics.

The application tracks metrics such as:

* Prediction request count
* Successful predictions
* Failed predictions
* Prediction latency
* Prediction distribution
* Endpoint health
* Connected endpoint information

Conceptually:

```text
FastAPI
   │
   ▼
Prometheus Metrics
   │
   ▼
Application Observability
```

---

# 📡 Production Data Capture

SageMaker Data Capture is enabled through the Endpoint Configuration.

Captured inference data is stored in Amazon S3.

```text
                     SageMaker Endpoint
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
           Prediction              Data Capture
                                        │
                                        ▼
                                        S3
```

The captured production data becomes an important input for production monitoring.

---

# 🔬 Production ML Monitoring

The monitoring strategy covers three major dimensions.

```text
                    Production System
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
       Data Quality    Data Drift    Model Quality
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
                      CloudWatch
                           │
                           ▼
                      SNS Alerting
```

---

# 🟢 Data Quality Monitoring

Data Quality Monitoring checks whether production data continues to satisfy expected constraints.

The monitoring workflow uses baseline information generated from reference data.

```text
Reference Data
      │
      ▼
   Baseline
      │
      ├──────────────┐
      ▼              ▼
 Statistics      Constraints
      │              │
      └──────┬───────┘
             │
             ▼
      Production Data
             │
             ▼
      Data Quality Monitor
             │
             ▼
        Violations
```

Data Quality Monitoring can identify issues such as:

* Missing values
* Unexpected values
* Constraint violations
* Statistical abnormalities
* Changes in feature behaviour

---

# 🟡 Data Drift Monitoring

Data Drift Monitoring identifies changes in production feature distributions compared with the baseline.

```text
Training / Reference Data
          │
          ▼
       Baseline
          │
          ▼
 Production Distribution
          │
          ▼
      Drift Analysis
          │
      ┌───┴───┐
      │       │
   Stable   Drift
```

Drift monitoring helps identify situations where real-world input data is changing relative to the data used to establish the baseline.

---

# 🔴 Model Quality Monitoring

Model Quality Monitoring focuses on whether the deployed model continues to provide acceptable predictions.

Conceptually:

```text
Production Input
      │
      ▼
Model Prediction
      │
      ▼
Actual Outcome
      │
      ▼
Model Quality Monitor
      │
      ▼
Performance Metrics
```

The model quality monitoring layer can evaluate metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

This complements data monitoring by answering a different question:

```text
Data Quality:
"Is the data healthy?"

Data Drift:
"Has the data distribution changed?"

Model Quality:
"Is the model still performing?"
```

---

# 📊 Monitoring Comparison

| Monitoring Type | Purpose                                    |
| --------------- | ------------------------------------------ |
| Data Quality    | Detect invalid or abnormal production data |
| Data Drift      | Detect changes in feature distributions    |
| Model Quality   | Detect degradation in model performance    |

Together these provide a broader production monitoring strategy.

---

# ☁️ CloudWatch

Amazon CloudWatch is used for operational monitoring and alerting.

The monitoring architecture is:

```text
SageMaker
    │
    ▼
CloudWatch Metrics
    │
    ▼
CloudWatch Alarm
```

CloudWatch can be used to monitor:

* Endpoint behaviour
* Inference metrics
* Errors
* Latency
* Monitoring violations
* Production health

---

# 🔔 SNS Alerting

CloudWatch alarms can trigger Amazon SNS notifications.

```text
Monitoring Metric
       │
       ▼
   CloudWatch
       │
       ▼
Threshold Breach
       │
       ▼
CloudWatch Alarm
       │
       ▼
      SNS
       │
       ▼
 Notification
```

This provides an operational alerting mechanism when monitoring conditions require attention.

---

# 🔄 CI/CD

The project separates training and deployment workflows using GitHub Actions.

---

## 🧪 Training CI/CD

Workflow:

```text
train.yml
```

Conceptual flow:

```text
Git Push
   │
   ▼
GitHub Actions
   │
   ▼
SageMaker Pipeline
   │
   ├── Processing
   ├── Training
   ├── Evaluation
   ├── Validation
   └── Model Registration
```

The training workflow produces versioned model candidates.

---

# 🚀 Deployment CI/CD

Workflow:

```text
deploy.yml
```

Conceptual flow:

```text
GitHub Actions
      │
      ▼
Latest Approved Model
      │
      ▼
Deployment Guard
      │
      ▼
SageMaker Model
      │
      ▼
Endpoint Configuration
      │
      ▼
Create / Update Endpoint
      │
      ▼
Wait for InService
```

The deployment workflow consumes the approved model rather than retraining the model.

---

# 🧠 Why Training and Deployment Are Separate

Separating the two workflows provides a clean production boundary.

```text
TRAINING
   │
   ▼
Model Candidate
   │
   ▼
Evaluation
   │
   ▼
Model Registry
   │
   ▼
Approval
   │
   ▼
DEPLOYMENT
   │
   ▼
Production
```

This means:

> Training a new model does not automatically mean that model becomes production.

The deployment process explicitly retrieves an approved model.

---

# 🛡️ Deployment Reliability

The deployment implementation includes multiple safeguards.

### Latest Approved Model

Only approved Model Package versions are considered for deployment.

### Model Reuse

Existing SageMaker Models are reused when appropriate.

### Endpoint Comparison

The deployment checks which model is currently serving.

### Idempotency

If the endpoint already serves the desired model, the deployment exits without making unnecessary changes.

### Immutable Endpoint Configurations

Each deployment creates a new Endpoint Configuration.

### Deployment Waiter

The workflow waits for the endpoint to reach:

```text
InService
```

before reporting success.

---

# 📁 Repository Structure

```text
diabetes-mlops/
│
├── pipelines/
│   ├── training_pipeline.py
│   └── deployment_pipeline.py
│
├── processing/
│   ├── preprocessing.py
│   └── evaluate.py
│
├── training/
│   └── train.py
│
├── inference/
│   └── inference.py
│
├── deployment/
│   ├── create_model.py
│   ├── create_endpoint_config.py
│   ├── deploy_endpoint.py
│   ├── retrieve_model.py
│   └── deployment_utils.py
│
├── monitoring/
│   ├── create_baseline.py
│   ├── create_monitor_schedule.py
│   ├── generate_test_traffic.py
│   └── inspect_monitoring.py
│
├── fastapp/
│   ├── app.py
│   ├── predict.py
│   ├── metrics.py
│   │
│   ├── schema/
│   │
│   ├── templates/
│   │
│   └── static/
│
├── config.py
│
├── train.yml
├── deploy.yml
│
└── README.md
```

---

# ⚙️ Configuration Management

Central project configuration is maintained through:

```text
config.py
```

Typical configuration values include:

```text
AWS Region
S3 Bucket
Raw Data Location
Model Package Group
Model Name Prefix
Endpoint Name
Execution Role ARN
Instance Type
Instance Count
Data Capture Location
Monitoring Location
CloudWatch Configuration
SNS Configuration
```

Keeping these values centralized avoids scattering environment-specific configuration throughout the codebase.

---

# ☁️ AWS Services Used

| AWS Service               | Responsibility                                              |
| ------------------------- | ----------------------------------------------------------- |
| Amazon S3                 | Data, model artifacts, captured data and monitoring outputs |
| SageMaker Processing      | Data preprocessing and evaluation                           |
| SageMaker Training        | Model training                                              |
| SageMaker Pipelines       | ML workflow orchestration                                   |
| SageMaker Model Registry  | Model versioning and approval                               |
| SageMaker Model           | Deployable model resource                                   |
| SageMaker Endpoint Config | Inference configuration                                     |
| SageMaker Endpoint        | Real-time inference                                         |
| SageMaker Model Monitor   | Production ML monitoring                                    |
| Amazon CloudWatch         | Metrics and alarms                                          |
| Amazon SNS                | Notifications                                               |
| AWS IAM                   | Access control                                              |

---

# 🧰 Technology Stack

| Category               | Technology               |
| ---------------------- | ------------------------ |
| Language               | Python                   |
| Machine Learning       | Scikit-Learn             |
| Cloud                  | AWS                      |
| ML Platform            | Amazon SageMaker         |
| Object Storage         | Amazon S3                |
| Pipeline               | SageMaker Pipelines      |
| Model Registry         | SageMaker Model Registry |
| API                    | FastAPI                  |
| Validation             | Pydantic                 |
| Application Metrics    | Prometheus               |
| AWS SDK                | Boto3                    |
| CI/CD                  | GitHub Actions           |
| Monitoring             | SageMaker Model Monitor  |
| Operational Monitoring | Amazon CloudWatch        |
| Alerting               | Amazon SNS               |

---

# 📋 Implementation Status

| Component                       | Status |
| ------------------------------- | :----: |
| Data Processing                 |    ✅   |
| Data Cleaning                   |    ✅   |
| Train / Validation / Test Split |    ✅   |
| SageMaker Processing Jobs       |    ✅   |
| SageMaker Training Jobs         |    ✅   |
| Model Evaluation                |    ✅   |
| Condition Step                  |    ✅   |
| Property Files                  |    ✅   |
| Model Metrics                   |    ✅   |
| SageMaker Model Registry        |    ✅   |
| Model Package Versioning        |    ✅   |
| Model Approval                  |    ✅   |
| Latest Approved Model Retrieval |    ✅   |
| SageMaker Model Creation        |    ✅   |
| SageMaker Model Reuse           |    ✅   |
| Endpoint State Validation       |    ✅   |
| Deployment Guard                |    ✅   |
| Endpoint Configuration          |    ✅   |
| Data Capture                    |    ✅   |
| Endpoint Create / Update        |    ✅   |
| FastAPI                         |    ✅   |
| Pydantic Validation             |    ✅   |
| Prometheus Metrics              |    ✅   |
| Data Quality Monitoring         |    ✅   |
| Data Drift Monitoring           |    ✅   |
| Model Quality Monitoring        |    ✅   |
| CloudWatch Monitoring           |    ✅   |
| CloudWatch Alarm                |    ✅   |
| SNS Alerting                    |    ✅   |
| GitHub Actions Training         |    ✅   |
| GitHub Actions Deployment       |    ✅   |
| Blue/Green Foundation           |    ✅   |
| Canary Foundation               |    ✅   |
| Automated Traffic Shifting      |   🔄   |
| Automated Rollback              |   🔄   |
| Automated Retraining            |   🔄   |

---

# 🧩 Important Architectural Concepts

## Model Registry

The Model Registry is the governance boundary.

```text
Training
   ↓
Evaluation
   ↓
Registration
   ↓
Approval
   ↓
Deployment
```

---

## SageMaker Model

A SageMaker Model is the deployable AWS resource.

```text
Model Package
      ↓
SageMaker Model
```

---

## Endpoint Configuration

An Endpoint Configuration describes how the model should run.

```text
SageMaker Model
      +
Instance Type
      +
Instance Count
      +
Traffic Configuration
      +
Data Capture
      ↓
Endpoint Configuration
```

---

## Endpoint

The Endpoint is the live production inference service.

```text
Endpoint Configuration
          ↓
      Endpoint
          ↓
   Real-Time Prediction
```

---

# 🔐 Production Design Principles

This project follows several important production MLOps principles.

### Separation of Concerns

Training, deployment, inference and monitoring are separate components.

### Model Governance

Production deployment is based on approved model versions.

### Infrastructure State Awareness

Deployment code checks the existing AWS state before making changes.

### Idempotent Deployment

Repeated execution should not create unnecessary resources when the desired state already exists.

### Immutable Deployment Configuration

New Endpoint Configurations are created rather than modifying existing configurations.

### Production Observability

Both application-level and ML-level monitoring are implemented.

### Controlled Promotion

The Model Registry acts as the promotion boundary between training and production.

---

# 🔁 End-to-End Production Flow

```text
                         ┌───────────────┐
                         │    Raw Data   │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │  Processing   │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    Training   │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   Evaluation  │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   Validation  │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Model Registry│
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    Approval   │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   Deployment  │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   SageMaker   │
                         │    Endpoint   │
                         └───────┬───────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
               ┌──────────┐             ┌───────────┐
               │ FastAPI  │             │ Data S3   │
               │   API    │             │  Capture  │
               └────┬─────┘             └─────┬─────┘
                    │                         │
                    ▼                         ▼
               Prometheus              SageMaker Monitor
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       │                      │                      │
                       ▼                      ▼                      ▼
                 Data Quality            Data Drift            Model Quality
                       │                      │                      │
                       └──────────────────────┼──────────────────────┘
                                              │
                                              ▼
                                         CloudWatch
                                              │
                                              ▼
                                             SNS
```

---

# 🧠 What This Project Demonstrates

This project goes beyond a traditional machine learning notebook.

A traditional ML workflow might look like:

```text
Train Model
    ↓
Save Model
    ↓
Deploy Model
```

This project implements a production-oriented lifecycle:

```text
Data
 ↓
Processing
 ↓
Training
 ↓
Evaluation
 ↓
Quality Gate
 ↓
Model Registry
 ↓
Approval
 ↓
Deployment
 ↓
Real-Time Serving
 ↓
Data Capture
 ↓
Data Quality
 ↓
Data Drift
 ↓
Model Quality
 ↓
CloudWatch
 ↓
Alerting
```

This demonstrates practical understanding of:

* Machine Learning
* MLOps
* AWS SageMaker
* Model Governance
* CI/CD
* Production Deployment
* API Engineering
* Observability
* Model Monitoring
* Data Monitoring
* Cloud Operations

---

# 🚀 Future Enhancements

The current implementation provides the foundation for additional production capabilities.

Potential future enhancements include:

* Automated Canary traffic shifting
* Automated Blue/Green promotion
* Automated rollback
* Endpoint auto scaling
* API Gateway integration
* Authentication and authorization
* CloudWatch dashboards
* Infrastructure as Code
* Terraform
* CloudFormation
* Automated retraining
* Automated model promotion
* Advanced deployment approvals
* Multi-environment deployment
* Development / Staging / Production environments

---

# 🎓 Learning Outcomes

This project demonstrates practical implementation of:

### Machine Learning

* Data preprocessing
* Feature preparation
* Model training
* Model evaluation
* Model validation

### MLOps

* SageMaker Processing Jobs
* SageMaker Training Jobs
* SageMaker Pipelines
* Condition Steps
* Property Files
* Model Metrics
* Model Registry
* Model Packages
* Model Approval
* Deployment Automation
* Data Capture
* Data Quality Monitoring
* Data Drift Monitoring
* Model Quality Monitoring

### AWS

* Amazon S3
* Amazon SageMaker
* AWS IAM
* Amazon CloudWatch
* Amazon SNS
* Boto3

### DevOps

* GitHub Actions
* CI/CD
* Deployment automation
* Configuration management
* State-aware deployment

### Application Engineering

* FastAPI
* Pydantic
* REST APIs
* Prometheus
* Production logging

---

# 🏆 Final Architecture Summary

```text
                 MACHINE LEARNING LIFECYCLE
                 ==========================

                         DATA
                          │
                          ▼
                     PROCESSING
                          │
                          ▼
                      TRAINING
                          │
                          ▼
                     EVALUATION
                          │
                          ▼
                    QUALITY GATE
                          │
                          ▼
                   MODEL REGISTRY
                          │
                          ▼
                       APPROVAL
                          │
                          ▼
                     DEPLOYMENT
                          │
                          ▼
                  SAGEMAKER ENDPOINT
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
              FastAPI         Data Capture
                 │                 │
                 ▼                 ▼
            Prometheus       S3 Production Data
                                   │
                                   ▼
                            Model Monitoring
                                   │
                     ┌─────────────┼─────────────┐
                     │             │             │
                     ▼             ▼             ▼
                Data Quality   Data Drift   Model Quality
                     │             │             │
                     └─────────────┼─────────────┘
                                   │
                                   ▼
                              CloudWatch
                                   │
                                   ▼
                                  SNS
```

---

# ⭐ Project Summary

This Diabetes Prediction project demonstrates a complete **production-oriented MLOps lifecycle using AWS SageMaker**.

The platform establishes a controlled path from:

```text
Model Development
       ↓
Model Validation
       ↓
Model Governance
       ↓
Production Deployment
       ↓
Real-Time Inference
       ↓
Production Monitoring
       ↓
Operational Alerting
```

The architecture is designed to provide **repeatability, traceability, deployment safety, model governance and production observability**.

---

<p align="center">

### 🩺 Diabetes Prediction MLOps Platform

**Train → Evaluate → Register → Approve → Deploy → Serve → Monitor → Alert**

Built with **Python • AWS SageMaker • FastAPI • GitHub Actions • Prometheus**

</p>

