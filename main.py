from prediction import prediction_pipeline
from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get('/')
def read():
    return {"prediction_model_name": "FOOD_101_categorical_image_classification"}


@app.post("/docs")
async def image_read_pred():
    Prediction_pipeline = prediction_pipeline()
    return Prediction_pipeline


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
