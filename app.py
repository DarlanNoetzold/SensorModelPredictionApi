from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import numpy as np

app = FastAPI()

# Carregar os modelos
models = {
    'cpu_usage_random_forest': load('/app/models/cpu_usage_RandomForest_model.joblib'),
    'cpu_usage_xgboost': load('/app/models/cpu_usage_XGBoost_model.joblib'),
    'cpu_usage_lightgbm': load('/app/models/cpu_usage_LightGBM_model.joblib'),
    'cpu_usage_catboost': load('/app/models/cpu_usage_CatBoost_model.joblib'),
    'cpu_usage_linear_regression': load('/app/models/cpu_usage_LinearRegression_model.joblib'),
    'cpu_usage_svr': load('/app/models/cpu_usage_SVR_model.joblib'),
    'cpu_usage_knn': load('/app/models/cpu_usage_KNN_model.joblib'),
    'cpu_usage_mlp': load('/app/models/cpu_usage_MLP_model.joblib'),

    'memory_usage_random_forest': load('/app/models/memory_usage_RandomForest_model.joblib'),
    'memory_usage_xgboost': load('/app/models/memory_usage_XGBoost_model.joblib'),
    'memory_usage_lightgbm': load('/app/models/memory_usage_LightGBM_model.joblib'),
    'memory_usage_catboost': load('/app/models/memory_usage_CatBoost_model.joblib'),
    'memory_usage_linear_regression': load('/app/models/memory_usage_LinearRegression_model.joblib'),
    'memory_usage_svr': load('/app/models/memory_usage_SVR_model.joblib'),
    'memory_usage_knn': load('/app/models/memory_usage_KNN_model.joblib'),
    'memory_usage_mlp': load('/app/models/memory_usage_MLP_model.joblib'),

    'thread_count_random_forest': load('/app/models/thread_count_RandomForest_model.joblib'),
    'thread_count_xgboost': load('/app/models/thread_count_XGBoost_model.joblib'),
    'thread_count_lightgbm': load('/app/models/thread_count_LightGBM_model.joblib'),
    'thread_count_catboost': load('/app/models/thread_count_CatBoost_model.joblib'),
    'thread_count_linear_regression': load('/app/models/thread_count_LinearRegression_model.joblib'),
    'thread_count_svr': load('/app/models/thread_count_SVR_model.joblib'),
    'thread_count_knn': load('/app/models/thread_count_KNN_model.joblib'),
    'thread_count_mlp': load('/app/models/thread_count_MLP_model.joblib'),

    'total_data_received_random_forest': load('/app/models/total_data_received_RandomForest_model.joblib'),
    'total_data_received_xgboost': load('/app/models/total_data_received_XGBoost_model.joblib'),
    'total_data_received_lightgbm': load('/app/models/total_data_received_LightGBM_model.joblib'),
    'total_data_received_catboost': load('/app/models/total_data_received_CatBoost_model.joblib'),
    'total_data_received_linear_regression': load('/app/models/total_data_received_LinearRegression_model.joblib'),
    'total_data_received_svr': load('/app/models/total_data_received_SVR_model.joblib'),
    'total_data_received_knn': load('/app/models/total_data_received_KNN_model.joblib'),
    'total_data_received_mlp': load('/app/models/total_data_received_MLP_model.joblib'),

    'total_data_filtered_random_forest': load('/app/models/total_data_filtered_RandomForest_model.joblib'),
    'total_data_filtered_xgboost': load('/app/models/total_data_filtered_XGBoost_model.joblib'),
    'total_data_filtered_lightgbm': load('/app/models/total_data_filtered_LightGBM_model.joblib'),
    'total_data_filtered_catboost': load('/app/models/total_data_filtered_CatBoost_model.joblib'),
    'total_data_filtered_linear_regression': load('/app/models/total_data_filtered_LinearRegression_model.joblib'),
    'total_data_filtered_svr': load('/app/models/total_data_filtered_SVR_model.joblib'),
    'total_data_filtered_knn': load('/app/models/total_data_filtered_KNN_model.joblib'),
    'total_data_filtered_mlp': load('/app/models/total_data_filtered_MLP_model.joblib'),

    'total_data_compressed_random_forest': load('/app/models/total_data_compressed_RandomForest_model.joblib'),
    'total_data_compressed_xgboost': load('/app/models/total_data_compressed_XGBoost_model.joblib'),
    'total_data_compressed_lightgbm': load('/app/models/total_data_compressed_LightGBM_model.joblib'),
    'total_data_compressed_catboost': load('/app/models/total_data_compressed_CatBoost_model.joblib'),
    'total_data_compressed_linear_regression': load('/app/models/total_data_compressed_LinearRegression_model.joblib'),
    'total_data_compressed_svr': load('/app/models/total_data_compressed_SVR_model.joblib'),
    'total_data_compressed_knn': load('/app/models/total_data_compressed_KNN_model.joblib'),
    'total_data_compressed_mlp': load('/app/models/total_data_compressed_MLP_model.joblib'),

    'total_data_aggregated_random_forest': load('/app/models/total_data_aggregated_RandomForest_model.joblib'),
    'total_data_aggregated_xgboost': load('/app/models/total_data_aggregated_XGBoost_model.joblib'),
    'total_data_aggregated_lightgbm': load('/app/models/total_data_aggregated_LightGBM_model.joblib'),
    'total_data_aggregated_catboost': load('/app/models/total_data_aggregated_CatBoost_model.joblib'),
    'total_data_aggregated_linear_regression': load('/app/models/total_data_aggregated_LinearRegression_model.joblib'),
    'total_data_aggregated_svr': load('/app/models/total_data_aggregated_SVR_model.joblib'),
    'total_data_aggregated_knn': load('/app/models/total_data_aggregated_KNN_model.joblib'),
    'total_data_aggregated_mlp': load('/app/models/total_data_aggregated_MLP_model.joblib'),

    'total_data_after_heuristics_random_forest': load('/app/models/total_data_after_heuristics_RandomForest_model.joblib'),
    'total_data_after_heuristics_xgboost': load('/app/models/total_data_after_heuristics_XGBoost_model.joblib'),
    'total_data_after_heuristics_lightgbm': load('/app/models/total_data_after_heuristics_LightGBM_model.joblib'),
    'total_data_after_heuristics_catboost': load('/app/models/total_data_after_heuristics_CatBoost_model.joblib'),
    'total_data_after_heuristics_linear_regression': load('/app/models/total_data_after_heuristics_LinearRegression_model.joblib'),
    'total_data_after_heuristics_svr': load('/app/models/total_data_after_heuristics_SVR_model.joblib'),
    'total_data_after_heuristics_knn': load('/app/models/total_data_after_heuristics_KNN_model.joblib'),
    'total_data_after_heuristics_mlp': load('/app/models/total_data_after_heuristics_MLP_model.joblib'),
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
