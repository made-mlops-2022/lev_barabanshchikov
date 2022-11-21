import logging
import os

import gdown
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from utils import extract_model, extract_data
from validation import InputData, OutputData

logger = logging.getLogger()

app = FastAPI()

model = None


@app.get("/")
def main():
    return "Welcome to Heart Disease Predictor! It is running and ready."


@app.on_event("startup")
def load_model():
    model_url = os.getenv("MODEL_URL")
    logger.info(f"Downloading model from {model_url}")
    # gdown.download_folder(url=model_url, quiet=True, output="online_inference/model")
    global model
    model = extract_model()
    logger.info("Model is loaded")


@app.post("/predict")
def predict(request: InputData):
    data = extract_data(request)
    logger.info("Data has been extracted")
    try:
        predictions = model.predict(data)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong during prediction"
        )
    logger.info("Prediction successful")
    return OutputData(predicted=[str(pred) for pred in predictions])


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model undefined"
        )
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"detail": "Model is ready. Everything is OK"})
    )


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exception: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content=jsonable_encoder({"detail": exception.errors(), "body": exception.body})
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
