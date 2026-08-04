---

# Diabetes Prediction MLOps Pipeline on AWS SageMaker

End-to-end MLOps pipeline for Diabetes Prediction built using **Amazon SageMaker Pipelines**, **Model Registry**, **Endpoint Deployment**, and **FastAPI**.

The project demonstrates how to build a production-ready machine learning system that automatically preprocesses data, trains a model, evaluates it, registers approved models, deploys them to a SageMaker Endpoint, and serves predictions through a FastAPI application.

---

# Architecture

```text
                   Raw Data (S3)
                         │
                         ▼
              SageMaker Processing Job
          (Preprocessing / Train-Test Split)
                         │
                         ▼
              SageMaker Training Job
             (Scikit-Learn Logistic Regression)
                         │
                         ▼
              SageMaker Processing Job
                 (Model Evaluation)
                         │
                         ▼
               Evaluation Metrics
        (accuracy, precision, recall, f1)
                         │
                         ▼
                Condition Step
           (Performance Threshold Check)
                         │
          ┌──────────────┴───────────────┐
          │                              │
       Pass                           Fail
          │                              │
          ▼                              ▼
 Register Model Package             Pipeline Stops
   (Model Registry)
          │
          ▼
 Approved Model Package
          │
          ▼
 Deployment Pipeline
(Create Model → Endpoint Config → Endpoint)
          │
          ▼
   SageMaker Real-Time Endpoint
          │
          ▼
        FastAPI
          │
          ▼
      Web / REST API
```

---

# Project Features

* SageMaker Pipelines
* Data preprocessing using Processing Jobs
* Scikit-Learn training jobs
* Automated model evaluation
* Conditional model registration
* SageMaker Model Registry
* Automated endpoint deployment
* FastAPI prediction service
* Pydantic request validation
* Prometheus metrics
* Endpoint health monitoring
* Production logging
* AWS SDK (Boto3)

---

# Technology Stack

| Category       | Technology               |
| -------------- | ------------------------ |
| Language       | Python                   |
| ML Framework   | Scikit-Learn             |
| Cloud          | AWS SageMaker            |
| Storage        | Amazon S3                |
| Model Registry | SageMaker Model Registry |
| Deployment     | SageMaker Endpoint       |
| API            | FastAPI                  |
| Validation     | Pydantic                 |
| Monitoring     | Prometheus               |
| AWS SDK        | boto3                    |

---

# Pipeline Workflow

## 1. Data Preprocessing

Processing Job

* Reads dataset from Amazon S3
* Cleans missing values
* Splits data into

```
Train
Validation
Test
```

Outputs are automatically stored in S3 by SageMaker Pipelines.

---

## 2. Model Training

Training Job

* Reads Train dataset
* Trains Logistic Regression model
* Saves model artifacts to Amazon S3

---

## 3. Model Evaluation

Processing Job

Evaluates:

* Accuracy
* Precision
* Recall
* F1 Score

Produces

```
evaluation.json
```

---

## 4. Model Validation

Pipeline Condition Step

Registers model only if

```
Accuracy ≥ 0.70
Precision ≥ 0.55
Recall ≥ 0.80
F1 ≥ 0.65
```

Otherwise the pipeline stops.

---

## 5. Model Registration

Approved models are registered in

```
DiabetesPredictionModel
```

using SageMaker Model Registry.

---

## 6. Deployment Pipeline

Deployment automatically

```
Retrieve latest Approved Model

↓

Create SageMaker Model

↓

Create Endpoint Configuration

↓

Create / Update Endpoint

↓

Zero downtime deployment
```

The deployment always uses the **latest approved Model Package**, ensuring production consistency.

---

# FastAPI Workflow

Prediction flow

```text
Client

↓

FastAPI

↓

Pydantic Validation

↓

predict.py

↓

SageMaker Endpoint

↓

inference.py

↓

Model Prediction

↓

FastAPI Response

↓

Client
```

---

# Request Example

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

# Response Example

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

# Monitoring

The FastAPI application exposes Prometheus metrics.

Collected metrics include

* Prediction Requests
* Successful Predictions
* Failed Predictions
* Prediction Distribution
* Prediction Latency
* Endpoint Health
* Connected Endpoint Information

---

# Health Endpoint

```
GET /health
```

Returns

```json
{
    "status": "OK",
    "endpoint": "diabetes-endpoint",
    "endpoint_status": "InService"
}
```

---

# Prediction Endpoint

```
POST /predict
```

Request

```json
{
    "pregnancies":6,
    "glucose":148,
    "blood_pressure":72,
    "skin_thickness":35,
    "insulin":1,
    "bmi":33.6,
    "diabetes_pedigree_function":0.627,
    "age":50
}
```

---

# Model Deployment Strategy

The deployment pipeline performs intelligent deployment.

* Retrieves the latest approved model
* Checks whether the endpoint already serves the latest version
* Reuses an existing SageMaker Model if available
* Creates a new Endpoint Configuration
* Updates the endpoint only when necessary

This avoids unnecessary deployments and supports model versioning.

---

# Repository Structure

```text
project/

├── pipelines/
│   ├── training_pipeline.py
│   ├── deployment_pipeline.py
│
├── processing/
│   ├── preprocessing.py
│   ├── evaluate.py
│
├── training/
│   ├── train.py
│
├── inference/
│   ├── inference.py
│
├── deployment/
│   ├── create_model.py
│   ├── create_endpoint_config.py
│   ├── deploy_endpoint.py
│   ├── retrieve_model.py
│   └── deployment_utils.py
│
├── fastapp/
│   ├── app.py
│   ├── predict.py
│   ├── metrics.py
│   ├── schema/
│   ├── templates/
│   ├── static/
│
├── config.py
```

---

# Learning Outcomes

This project demonstrates practical experience with

* SageMaker Processing Jobs
* SageMaker Training Jobs
* SageMaker Pipelines
* Condition Steps
* Property Files
* Model Metrics
* Model Registry
* Model Packages
* SageMaker Models
* Endpoint Configurations
* Real-Time Endpoints
* Automated Deployment
* FastAPI Integration
* Boto3 SDK
* Prometheus Monitoring
* Production-ready ML Inference

---

# Future Improvements

* CI/CD using GitHub Actions
* Infrastructure as Code (Terraform or CloudFormation)
* Model Monitoring
* Data Quality Monitoring
* Drift Detection
* Blue/Green Deployment
* Auto Scaling
* Authentication with API Gateway
* CloudWatch Dashboards
* Canary Deployments

---
