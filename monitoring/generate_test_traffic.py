import random
import requests
import uuid
import json
import time


FASTAPI_URL = "http://localhost:8000/predict"

GROUND_TRUTH_FILE = "ground_truth.jsonl"


ground_truth_records = []


for i in range(100):

    inference_id = str(uuid.uuid4())


    payload = {

        "pregnancies": random.randint(0, 10),

        "glucose": random.randint(80, 180),

        "blood_pressure": random.randint(60, 100),

        "skin_thickness": random.randint(15, 40),

        "insulin": random.randint(50, 200),

        "bmi": random.uniform(20, 40),

        "diabetes_pedigree_function":
            random.uniform(0.1, 1.0),

        "age": random.randint(20, 70),
    }


    # Demo only.
    # In real production this comes from
    # the actual outcome later.

    actual_label = random.randint(0, 1)


    response = requests.post(

        FASTAPI_URL,

        json=payload,

        headers={
            "X-Inference-Id": inference_id
        },
    )


    print(
        i + 1,
        inference_id,
        response.status_code,
        response.json(),
    )


    # Ground truth record
    ground_truth_record = {

        "groundTruthData": {

            "data": str(actual_label),

            "encoding": "CSV",
        },

        "eventMetadata": {

            "eventId": inference_id
        },

        "eventVersion": "0",
    }


    ground_truth_records.append(
        ground_truth_record
    )


    time.sleep(1)


with open(
    GROUND_TRUTH_FILE,
    "w"
) as f:

    for record in ground_truth_records:

        f.write(
            json.dumps(record)
            + "\n"
        )


print(
    f"Created {GROUND_TRUTH_FILE}"
)