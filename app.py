from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import numpy as np
import os

app = FastAPI()

# Verificação de inicialização da aplicação
print("Starting FastAPI application...")

# Carregar os modelos com tratamento de erros
models = {}

model_paths = {
    'cpu_usage_catboost': '/app/models/CPU Usage_CatBoost_model.joblib',
    'cpu_usage_knn': '/app/models/CPU Usage_KNN_model.joblib',
    'cpu_usage_lightgbm': '/app/models/CPU Usage_LightGBM_model.joblib',
    'cpu_usage_linear_regression': '/app/models/CPU Usage_LinearRegression_model.joblib',
    'cpu_usage_mlp': '/app/models/CPU Usage_MLP_model.joblib',
    'cpu_usage_random_forest': '/app/models/CPU Usage_RandomForest_model.joblib',
    'cpu_usage_svr': '/app/models/CPU Usage_SVR_model.joblib',
    'cpu_usage_xgboost': '/app/models/CPU Usage_XGBoost_model.joblib',

    'error_count_knn': '/app/models/Error Count_KNN_model.joblib',
    'error_count_lightgbm': '/app/models/Error Count_LightGBM_model.joblib',
    'error_count_linear_regression': '/app/models/Error Count_LinearRegression_model.joblib',
    'error_count_mlp': '/app/models/Error Count_MLP_model.joblib',
    'error_count_random_forest': '/app/models/Error Count_RandomForest_model.joblib',
    'error_count_svr': '/app/models/Error Count_SVR_model.joblib',
    'error_count_xgboost': '/app/models/Error Count_XGBoost_model.joblib',

    'memory_usage_catboost': '/app/models/Memory Usage_CatBoost_model.joblib',
    'memory_usage_knn': '/app/models/Memory Usage_KNN_model.joblib',
    'memory_usage_lightgbm': '/app/models/Memory Usage_LightGBM_model.joblib',
    'memory_usage_linear_regression': '/app/models/Memory Usage_LinearRegression_model.joblib',
    'memory_usage_mlp': '/app/models/Memory Usage_MLP_model.joblib',
    'memory_usage_random_forest': '/app/models/Memory Usage_RandomForest_model.joblib',
    'memory_usage_svr': '/app/models/Memory Usage_SVR_model.joblib',
    'memory_usage_xgboost': '/app/models/Memory Usage_XGBoost_model.joblib',

    'thread_count_catboost': '/app/models/Thread Count_CatBoost_model.joblib',
    'thread_count_knn': '/app/models/Thread Count_KNN_model.joblib',
    'thread_count_lightgbm': '/app/models/Thread Count_LightGBM_model.joblib',
    'thread_count_linear_regression': '/app/models/Thread Count_LinearRegression_model.joblib',
    'thread_count_mlp': '/app/models/Thread Count_MLP_model.joblib',
    'thread_count_random_forest': '/app/models/Thread Count_RandomForest_model.joblib',
    'thread_count_svr': '/app/models/Thread Count_SVR_model.joblib',
    'thread_count_xgboost': '/app/models/Thread Count_XGBoost_model.joblib',

    'total_data_after_heuristics_catboost': '/app/models/Total Data After Heuristics_CatBoost_model.joblib',
    'total_data_after_heuristics_knn': '/app/models/Total Data After Heuristics_KNN_model.joblib',
    'total_data_after_heuristics_lightgbm': '/app/models/Total Data After Heuristics_LightGBM_model.joblib',
    'total_data_after_heuristics_linear_regression': '/app/models/Total Data After Heuristics_LinearRegression_model.joblib',
    'total_data_after_heuristics_mlp': '/app/models/Total Data After Heuristics_MLP_model.joblib',
    'total_data_after_heuristics_random_forest': '/app/models/Total Data After Heuristics_RandomForest_model.joblib',
    'total_data_after_heuristics_svr': '/app/models/Total Data After Heuristics_SVR_model.joblib',
    'total_data_after_heuristics_xgboost': '/app/models/Total Data After Heuristics_XGBoost_model.joblib',

    'total_data_aggregated_catboost': '/app/models/Total Data Aggregated_CatBoost_model.joblib',
    'total_data_aggregated_knn': '/app/models/Total Data Aggregated_KNN_model.joblib',
    'total_data_aggregated_lightgbm': '/app/models/Total Data Aggregated_LightGBM_model.joblib',
    'total_data_aggregated_linear_regression': '/app/models/Total Data Aggregated_LinearRegression_model.joblib',
    'total_data_aggregated_mlp': '/app/models/Total Data Aggregated_MLP_model.joblib',
    'total_data_aggregated_random_forest': '/app/models/Total Data Aggregated_RandomForest_model.joblib',
    'total_data_aggregated_svr': '/app/models/Total Data Aggregated_SVR_model.joblib',
    'total_data_aggregated_xgboost': '/app/models/Total Data Aggregated_XGBoost_model.joblib',

    'total_data_compressed_catboost': '/app/models/Total Data Compressed_CatBoost_model.joblib',
    'total_data_compressed_knn': '/app/models/Total Data Compressed_KNN_model.joblib',
    'total_data_compressed_lightgbm': '/app/models/Total Data Compressed_LightGBM_model.joblib',
    'total_data_compressed_linear_regression': '/app/models/Total Data Compressed_LinearRegression_model.joblib',
    'total_data_compressed_mlp': '/app/models/Total Data Compressed_MLP_model.joblib',
    'total_data_compressed_random_forest': '/app/models/Total Data Compressed_RandomForest_model.joblib',
    'total_data_compressed_svr': '/app/models/Total Data Compressed_SVR_model.joblib',
    'total_data_compressed_xgboost': '/app/models/Total Data Compressed_XGBoost_model.joblib',

    'total_data_filtered_catboost': '/app/models/Total Data Filtered_CatBoost_model.joblib',
    'total_data_filtered_knn': '/app/models/Total Data Filtered_KNN_model.joblib',
    'total_data_filtered_lightgbm': '/app/models/Total Data Filtered_LightGBM_model.joblib',
    'total_data_filtered_linear_regression': '/app/models/Total Data Filtered_LinearRegression_model.joblib',
    'total_data_filtered_mlp': '/app/models/Total Data Filtered_MLP_model.joblib',
    'total_data_filtered_random_forest': '/app/models/Total Data Filtered_RandomForest_model.joblib',
    'total_data_filtered_svr': '/app/models/Total Data Filtered_SVR_model.joblib',
    'total_data_filtered_xgboost': '/app/models/Total Data Filtered_XGBoost_model.joblib',

    'total_data_received_catboost': '/app/models/Total Data Received_CatBoost_model.joblib',
    'total_data_received_knn': '/app/models/Total Data Received_KNN_model.joblib',
    'total_data_received_lightgbm': '/app/models/Total Data Received_LightGBM_model.joblib',
    'total_data_received_linear_regression': '/app/models/Total Data Received_LinearRegression_model.joblib',
    'total_data_received_mlp': '/app/models/Total Data Received_MLP_model.joblib',
    'total_data_received_random_forest': '/app/models/Total Data Received_RandomForest_model.joblib',
    'total_data_received_svr': '/app/models/Total Data Received_SVR_model.joblib',
    'total_data_received_xgboost': '/app/models/Total Data Received_XGBoost_model.joblib',
}

for model_name, model_path in model_paths.items():
    try:
        if not os.path.exists(model_path):
            print(f"Model file does not exist: {model_path}")
            continue
        models[model_name] = load(model_path)
        print(f"{model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")


class PredictionRequest(BaseModel):
    metric: str
    current_value: float
    num_predictions: int


@app.post("/predict/")
def predict_next_value(request: PredictionRequest):
    print(f"Received prediction request: {request.metric}")

    if request.metric not in models:
        raise HTTPException(status_code=404, detail="Metric not found")

    model = models[request.metric]
    predictions = []

    current_value = request.current_value

    for _ in range(request.num_predictions):
        try:
            next_value = model.predict(np.array([[current_value]]))[0]
            predictions.append(next_value)
            current_value = next_value
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"predictions": predictions}


# Verificação final de inicialização
print("FastAPI application started successfully.")