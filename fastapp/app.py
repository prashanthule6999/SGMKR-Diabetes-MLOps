# FastAPI application
import time
import logging
import uvicorn
from pathlib import Path
from fastapi import Request
from fastapi.responses import Response
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest
from fastapi.templating import Jinja2Templates
from fastapp.schema.user_input import User_Input
from fastapi import FastAPI, HTTPException
from fastapp.predict import (
    predict_output,
    check_endpoint,
)
from fastapp.schema.prediction_response import Prediction_Response
from fastapp.metrics import ENDPOINT_INFO, ENDPOINT_AVAILABLE, PREDICTION_REQUESTS, PREDICTION_SUCCESS, PREDICTION_FAILURE, PREDICTION_RESULT, PREDICTION_LATENCY
from config import ENDPOINT_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Startup / Shutdown
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        check_endpoint(ENDPOINT_NAME)

        ENDPOINT_AVAILABLE.set(1)

        ENDPOINT_INFO.info({
            "endpoint": ENDPOINT_NAME
        })

        logger.info(
            "Connected to SageMaker endpoint: %s",
            ENDPOINT_NAME,
        )

    except Exception:

        ENDPOINT_AVAILABLE.set(0)

        logger.exception(
            "Model loading failed"
        )

        raise

    yield

    logger.info("Application shutdown")


# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Diabetes Prediction API",
    version="1.0",
    lifespan=lifespan
)
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ------------------------------------------------------------------------------
# Home
# ------------------------------------------------------------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ------------------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------------------
@app.get("/health")
def health():

    try:
        endpoint_status = check_endpoint(ENDPOINT_NAME)

        ENDPOINT_AVAILABLE.set(1)

        return {
            "status": "OK",
            "endpoint": ENDPOINT_NAME,
            "endpoint_status": endpoint_status
        }

    except Exception as e:

        ENDPOINT_AVAILABLE.set(0)

        logger.exception("Health check failed.")

        raise HTTPException(
            status_code=503,
            detail={
                "status": "FAILED",
                "endpoint": ENDPOINT_NAME,
                "error": str(e)
            }
        )

# ------------------------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------------------------


@app.post(
    "/predict",
    response_model=Prediction_Response
)
def predict_diabetes(data: User_Input):

    PREDICTION_REQUESTS.inc()

    start_time = time.perf_counter()

    try:

        user_input = {
            "Pregnancies": data.pregnancies,
            "Glucose": data.glucose,
            "BloodPressure": data.blood_pressure,
            "SkinThickness": data.skin_thickness,
            "Insulin": data.insulin,
            "BMI": data.bmi,
            "DiabetesPedigreeFunction": data.diabetes_pedigree_function,
            "Age": data.age
        }

        response = predict_output(user_input)

        PREDICTION_SUCCESS.inc()

        prediction = response.get("prediction")

        if prediction == 1:

            PREDICTION_RESULT.labels(
                result="diabetes"
            ).inc()

        else:

            PREDICTION_RESULT.labels(
                result="no_diabetes"
            ).inc()

        return response

    except Exception as e:

        PREDICTION_FAILURE.inc()

        logger.exception("Prediction request failed.")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        PREDICTION_LATENCY.observe(
            time.perf_counter() - start_time
        )

# ------------------------------------------------------------------------------
# Prometheus Metrics Endpoint
# ------------------------------------------------------------------------------


@app.get(
    "/metrics",
    include_in_schema=False
)
def metrics():

    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000
    )
