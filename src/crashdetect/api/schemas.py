"""Request and response models of the API (shown in Swagger at /docs)."""

from pydantic import BaseModel, Field, model_validator

MIN_SAMPLES = 10
_EXAMPLE = {
    "ax": [0.1, 0.2, 0.1, 0.0, 0.1, 0.3, 0.2, 0.1, 0.0, 0.1],
    "ay": [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
    "az": [9.8, 9.7, 9.9, 9.8, 9.8, 9.6, 9.9, 9.8, 9.7, 9.8],
    "rx": [0.01, 0.02, 0.01, 0.0, 0.01, 0.02, 0.01, 0.0, 0.01, 0.02],
    "ry": [0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01],
    "rz": [0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.0],
}


class Ride(BaseModel):
    """IMU recording of one ride: 6 series of the same length (accelerometer + gyroscope)."""

    ax: list[float | None] = Field(description="Acceleration, x axis")
    ay: list[float | None] = Field(description="Acceleration, y axis")
    az: list[float | None] = Field(description="Acceleration, z axis")
    rx: list[float | None] = Field(description="Rotation rate, x axis")
    ry: list[float | None] = Field(description="Rotation rate, y axis")
    rz: list[float | None] = Field(description="Rotation rate, z axis")

    model_config = {"json_schema_extra": {"examples": [_EXAMPLE]}}

    @model_validator(mode="after")
    def same_length(self):
        lengths = {len(getattr(self, axis)) for axis in ("ax", "ay", "az", "rx", "ry", "rz")}
        if len(lengths) != 1:
            raise ValueError("ax, ay, az, rx, ry, rz must have the same length")
        if lengths.pop() < MIN_SAMPLES:
            raise ValueError(f"a ride needs at least {MIN_SAMPLES} samples")
        return self


class Prediction(BaseModel):
    crash_probability: float
    crash_detected: bool
    threshold: float
    n_samples: int
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    test_roc_auc: float | None = None


class Health(BaseModel):
    status: str
    model_loaded: bool
