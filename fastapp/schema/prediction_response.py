from typing import Annotated
from pydantic import BaseModel, Field


class Prediction_Response(BaseModel):

    prediction: Annotated[
        int,
        Field(
            description="Predicted class (0 = No Diabetes, 1 = Diabetes)",
            examples=[1],
        )
    ]

    diabetes_probability: Annotated[
        float,
        Field(
            description="Predicted probability of diabetes",
            examples=[0.8421],
        )
    ]

    threshold: Annotated[
        float,
        Field(
            description="Decision threshold used for classification",
            examples=[0.3015],
        )
    ]

    class_probabilities: dict[str, float] = Field(
        description="Probability distribution for each class",
        examples=[{
            "No Diabetes": 0.1579,
            "Diabetes": 0.8421,
        }],
    )
