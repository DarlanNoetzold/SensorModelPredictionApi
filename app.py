from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import numpy as np

app = FastAPI()

# Carregar os modelos
models = {
    'cpu_usage': load('/app/models/cpu_usage_model.joblib'),
    'memory_usage': load('/app/models/memory_usage_model.joblib'),
    'thread_count': load('/app/models/thread_count_model.joblib'),
    'total_data_received': load('/app/models/total_data_received_model.joblib'),
    'total_data_filtered': load('/app/models/total_data_filtered_model.joblib'),
    'total_data_compressed': load('/app/models/total_data_compressed_model.joblib'),
    'total_data_aggregated': load('/app/models/total_data_aggregated_model.joblib'),
    'total_data_after_heuristics': load('/app/models/total_data_after_heuristics_model.joblib'),
}

class PredictionRequest(BaseModel):
    metric: str
    current_value: float
    num_predictions: int

@app.post("/predict/")
def predict_next_value(request: PredictionRequest):
    if request.metric not in models:
        raise HTTPException(status_code=404, detail="Metric not found")

    model = models[request.metric]
    predictions = []

    current_value = request.current_value

    for _ in range(request.num_predictions):
        # Fazer a previsão
        next_value = model.predict(np.array([[current_value]]))[0]
        predictions.append(next_value)
        # Usar o valor previsto como entrada para a próxima previsão
        current_value = next_value

    return {"predictions": predictions}
