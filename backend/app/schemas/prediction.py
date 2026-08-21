from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Matches the features the model's ColumnTransformer was trained on."""

    location: str = Field(..., min_length=1, examples=["Whitefield, Bangalore"])
    carpet_area_sqft: float = Field(..., gt=0, examples=[1200.0])
    floor_num: int = Field(..., ge=-1, le=200, examples=[3])
    bathroom: int = Field(..., ge=0, le=20, examples=[2])
    balcony: int = Field(..., ge=0, le=20, examples=[1])
    furnishing: str = Field(..., examples=["Semi-Furnished"])
    transaction: str = Field(..., examples=["Resale"])
    ownership: str = Field(..., examples=["Freehold"])
    facing: str = Field(..., examples=["East"])

    @field_validator("location", "furnishing", "transaction", "ownership", "facing")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v.strip()


class PredictionResponse(BaseModel):
    predicted_price: float


class HealthResponse(BaseModel):
    status: str
