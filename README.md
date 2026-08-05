<div align="center">

# 🚀 End-to-End MLOps Pipeline on AWS SageMaker

### Production-Ready Machine Learning Pipeline for Diabetes Prediction

Build • Train • Evaluate • Register • Deploy • Serve • Monitor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![AWS](https://img.shields.io/badge/AWS-SageMaker-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub_Actions-blue)
![Prometheus](https://img.shields.io/badge/Monitoring-Prometheus-red)

</div>

---

# 📖 Project Overview

This project demonstrates an **end-to-end MLOps workflow** built using **Amazon SageMaker** for training, evaluating, registering, deploying, and serving a machine learning model for diabetes prediction.

The solution follows production-oriented MLOps practices by separating the **training pipeline** from the **deployment pipeline**, allowing only approved models to be promoted to production.

The deployed model is exposed through a **FastAPI** application, which validates incoming requests using **Pydantic**, forwards prediction requests to a **SageMaker Real-Time Endpoint**, and exposes **Prometheus metrics** for monitoring.

---

# 🎯 Objectives

- Build a reproducible machine learning pipeline
- Automate model training and evaluation
- Register only validated models
- Support manual approval before deployment
- Deploy approved models to a SageMaker Endpoint
- Serve predictions using FastAPI
- Monitor API health and performance
- Demonstrate production-ready MLOps practices

---

# 🏗️ Solution Architecture

```text
                     Amazon S3
                  (Raw Dataset)
                          │
                          ▼
          SageMaker Processing Job
             Data Preprocessing
                          │
                          ▼
          SageMaker Training Job
             Model Training
                          │
                          ▼
          SageMaker Processing Job
             Model Evaluation
                          │
                          ▼
              Condition Step
         (Performance Validation)
                          │
            ┌─────────────┴─────────────┐
            │                           │
         Pass                        Fail
            │                           │
            ▼                           ▼
 SageMaker Model Registry         Pipeline Stops
(PendingManualApproval)
            │
     Manual Approval
            │
            ▼
     Deployment Pipeline
            │
            ▼
     SageMaker Model
            │
            ▼
   Endpoint Configuration
            │
            ▼
 SageMaker Real-Time Endpoint
            │
            ▼
         FastAPI API
            │
            ▼
        Client / UI
```

---

# 🚀 Key Features

- ✅ SageMaker Processing Jobs
- ✅ SageMaker Training Jobs
- ✅ SageMaker Pipelines
- ✅ Automated Data Preprocessing
- ✅ Automated Model Evaluation
- ✅ Conditional Model Registration
- ✅ SageMaker Model Registry
- ✅ Manual Model Approval
- ✅ Automated Deployment Pipeline
- ✅ SageMaker Real-Time Endpoint
- ✅ FastAPI Prediction Service
- ✅ Pydantic Request Validation
- ✅ Prometheus Metrics
- ✅ Endpoint Health Checks
- ✅ Production Logging
- ✅ GitHub Actions CI/CD

---

# 🛠️ Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-Learn |
| Cloud Platform | AWS SageMaker |
| Data Storage | Amazon S3 |
| Processing | SageMaker Processing Jobs |
| Training | SageMaker Training Jobs |
| Pipeline Orchestration | SageMaker Pipelines |
| Model Registry | SageMaker Model Registry |
| Deployment | SageMaker Endpoints |
| API Framework | FastAPI |
| Validation | Pydantic |
| Monitoring | Prometheus |
| CI/CD | GitHub Actions |
| AWS SDK | boto3 |
| Version Control | Git |

---

# 🔄 Training Pipeline

The training pipeline is orchestrated using **SageMaker Pipelines**.

```text
Raw Dataset (S3)
        │
        ▼
Preprocessing Step
        │
        ▼
Training Step
        │
        ▼
Evaluation Step
        │
        ▼
Condition Step
        │
        ▼
Model Registry
```

## Pipeline Components

### Data Preprocessing

- Reads raw data from Amazon S3
- Cleans missing values
- Performs train/validation/test split
- Stores processed datasets automatically in S3

### Model Training

- Trains a Logistic Regression model
- Saves model artifacts
- Uploads artifacts to Amazon S3

### Model Evaluation

Calculates:

- Accuracy
- Precision
- Recall
- F1 Score

Produces:

```
evaluation.json
```

### Model Validation

The model is registered only if:

| Metric | Threshold |
|---------|----------:|
| Accuracy | ≥ 0.70 |
| Precision | ≥ 0.55 |
| Recall | ≥ 0.80 |
| F1 Score | ≥ 0.65 |

Otherwise, the pipeline terminates without registering the model.

---

# 📦 SageMaker Model Registry

Validated models are stored in a SageMaker Model Package Group.

```
DiabetesPredictionModel

├── Version 1
├── Version 2
├── Version 3
└── Version N
```

Each registered model is assigned the status:

```
PendingManualApproval
```

Only approved models are eligible for deployment.

---

# 🚀 Deployment Pipeline

The deployment pipeline performs intelligent deployment to avoid unnecessary endpoint updates.

```text
Retrieve Latest Approved Model
            │
            ▼
Endpoint Already Serving Model?
            │
     ┌──────┴──────┐
     │             │
    Yes           No
     │             │
     ▼             ▼
 Stop      Model Exists?
                   │
          ┌────────┴────────┐
          │                 │
        Yes                No
          │                 │
          ▼                 ▼
     Reuse Model     Create Model
             │
             ▼
Create Endpoint Configuration
             │
             ▼
Create / Update Endpoint
```

### Deployment Features

- Retrieves the latest approved model
- Reuses existing SageMaker Models when available
- Creates unique Endpoint Configurations
- Updates the endpoint only when required
- Supports model versioning

---

# 🌐 FastAPI Inference Workflow

```text
Client Request
      │
      ▼
FastAPI
      │
      ▼
Pydantic Validation
      │
      ▼
predict.py
      │
      ▼
SageMaker Runtime
      │
      ▼
Real-Time Endpoint
      │
      ▼
Inference Container
      │
      ▼
Prediction Response
      │
      ▼
FastAPI Response
```

---

# 📥 Sample Request

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

---

# 📤 Sample Response

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

# 📊 Monitoring

The FastAPI application exports Prometheus metrics.

Available metrics include:

- Prediction Requests
- Successful Predictions
- Failed Predictions
- Prediction Latency
- Prediction Distribution
- Model Status
- Endpoint Health

---

# 🔁 CI/CD Workflow

## Training Workflow

```text
Developer
     │
     ▼
Git Push
     │
     ▼
GitHub Actions
(train.yml)
     │
     ▼
run_pipeline.py
     │
     ▼
Training Pipeline
     │
     ▼
Model Registry
```

---

## Deployment Workflow

```text
Manual Approval
       │
       ▼
GitHub Actions
(deploy.yml)
       │
       ▼
run_deployment.py
       │
       ▼
Deployment Pipeline
       │
       ▼
SageMaker Endpoint
```

---

# 📁 Repository Structure

```text
.
├── processing/
├── training/
├── inference/
├── deployment/
├── fastapp/
│   ├── schema/
│   ├── static/
│   ├── templates/
│   ├── app.py
│   ├── predict.py
│   └── metrics.py
│
├── .github/
│   └── workflows/
│       ├── train.yml
│       └── deploy.yml
│
├── pipelines/
│   ├── training_pipeline.py
│   ├── deployment_pipeline.py
│   ├── run_pipeline.py
│   ├── run_deployment.py
├── config.py
└── README.md
```

---

# 🧠 Key Concepts Demonstrated

- SageMaker Processing Jobs
- SageMaker Training Jobs
- SageMaker Pipelines
- ProcessingStep
- TrainingStep
- ConditionStep
- Property Files
- Model Metrics
- SageMaker Model Registry
- Model Versioning
- Manual Approval Workflow
- SageMaker Model
- Endpoint Configuration
- SageMaker Real-Time Endpoints
- FastAPI Integration
- Pydantic Validation
- Boto3 SDK
- Prometheus Monitoring
- GitHub Actions CI/CD

---

# 🚧 Future Enhancements

- SageMaker Model Monitor
- CloudWatch Dashboard
- CloudWatch Alarms
- SNS Notifications
- Blue/Green Deployment
- Infrastructure as Code (Terraform / CloudFormation)
- API Gateway Authentication
- Auto Scaling
- EventBridge + Lambda-based Deployment Automation

---

# 📸 Screenshots

> Add screenshots of the following components:

- SageMaker Pipeline
- Processing Job
- Training Job
- Evaluation Job
- Model Registry
- Endpoint
- FastAPI Swagger UI
- GitHub Actions Workflow
- Prometheus Metrics

---

# 👨‍💻 Author

**Prashant Hule**

**MLOps Engineer | AWS SageMaker | Python | FastAPI | CI/CD | Docker | Kubernetes | Production Machine Learning**

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
