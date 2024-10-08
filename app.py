from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Carregar os modelos
models = {
    'cpu_usage_catboost': load('/app/models/CPU Usage_CatBoost_model.joblib'),
    'cpu_usage_knn': load('/app/models/CPU Usage_KNN_model.joblib'),
    'cpu_usage_lightgbm': load('/app/models/CPU Usage_LightGBM_model.joblib'),
    'cpu_usage_linear_regression': load('/app/models/CPU Usage_LinearRegression_model.joblib'),
    'cpu_usage_mlp': load('/app/models/CPU Usage_MLP_model.joblib'),
    'cpu_usage_random_forest': load('/app/models/CPU Usage_RandomForest_model.joblib'),
    'cpu_usage_svr': load('/app/models/CPU Usage_SVR_model.joblib'),
    'cpu_usage_xgboost': load('/app/models/CPU Usage_XGBoost_model.joblib'),

    'error_count_knn': load('/app/models/Error Count_KNN_model.joblib'),
    'error_count_lightgbm': load('/app/models/Error Count_LightGBM_model.joblib'),
    'error_count_linear_regression': load('/app/models/Error Count_LinearRegression_model.joblib'),
    'error_count_mlp': load('/app/models/Error Count_MLP_model.joblib'),
    'error_count_random_forest': load('/app/models/Error Count_RandomForest_model.joblib'),
    'error_count_svr': load('/app/models/Error Count_SVR_model.joblib'),
    'error_count_xgboost': load('/app/models/Error Count_XGBoost_model.joblib'),

    'memory_usage_catboost': load('/app/models/Memory Usage_CatBoost_model.joblib'),
    'memory_usage_knn': load('/app/models/Memory Usage_KNN_model.joblib'),
    'memory_usage_lightgbm': load('/app/models/Memory Usage_LightGBM_model.joblib'),
    'memory_usage_linear_regression': load('/app/models/Memory Usage_LinearRegression_model.joblib'),
    'memory_usage_mlp': load('/app/models/Memory Usage_MLP_model.joblib'),
    'memory_usage_random_forest': load('/app/models/Memory Usage_RandomForest_model.joblib'),
    'memory_usage_svr': load('/app/models/Memory Usage_SVR_model.joblib'),
    'memory_usage_xgboost': load('/app/models/Memory Usage_XGBoost_model.joblib'),

    'thread_count_catboost': load('/app/models/Thread Count_CatBoost_model.joblib'),
    'thread_count_knn': load('/app/models/Thread Count_KNN_model.joblib'),
    'thread_count_lightgbm': load('/app/models/Thread Count_LightGBM_model.joblib'),
    'thread_count_linear_regression': load('/app/models/Thread Count_LinearRegression_model.joblib'),
    'thread_count_mlp': load('/app/models/Thread Count_MLP_model.joblib'),
    'thread_count_random_forest': load('/app/models/Thread Count_RandomForest_model.joblib'),
    'thread_count_svr': load('/app/models/Thread Count_SVR_model.joblib'),
    'thread_count_xgboost': load('/app/models/Thread Count_XGBoost_model.joblib'),

    'total_data_after_heuristics_catboost': load('/app/models/Total Data After Heuristics_CatBoost_model.joblib'),
    'total_data_after_heuristics_knn': load('/app/models/Total Data After Heuristics_KNN_model.joblib'),
    'total_data_after_heuristics_lightgbm': load('/app/models/Total Data After Heuristics_LightGBM_model.joblib'),
    'total_data_after_heuristics_linear_regression': load('/app/models/Total Data After Heuristics_LinearRegression_model.joblib'),
    'total_data_after_heuristics_mlp': load('/app/models/Total Data After Heuristics_MLP_model.joblib'),
    'total_data_after_heuristics_random_forest': load('/app/models/Total Data After Heuristics_RandomForest_model.joblib'),
    'total_data_after_heuristics_svr': load('/app/models/Total Data After Heuristics_SVR_model.joblib'),
    'total_data_after_heuristics_xgboost': load('/app/models/Total Data After Heuristics_XGBoost_model.joblib'),

    'total_data_aggregated_catboost': load('/app/models/Total Data Aggregated_CatBoost_model.joblib'),
    'total_data_aggregated_knn': load('/app/models/Total Data Aggregated_KNN_model.joblib'),
    'total_data_aggregated_lightgbm': load('/app/models/Total Data Aggregated_LightGBM_model.joblib'),
    'total_data_aggregated_linear_regression': load('/app/models/Total Data Aggregated_LinearRegression_model.joblib'),
    'total_data_aggregated_mlp': load('/app/models/Total Data Aggregated_MLP_model.joblib'),
    'total_data_aggregated_random_forest': load('/app/models/Total Data Aggregated_RandomForest_model.joblib'),
    'total_data_aggregated_svr': load('/app/models/Total Data Aggregated_SVR_model.joblib'),
    'total_data_aggregated_xgboost': load('/app/models/Total Data Aggregated_XGBoost_model.joblib'),

    'total_data_compressed_catboost': load('/app/models/Total Data Compressed_CatBoost_model.joblib'),
    'total_data_compressed_knn': load('/app/models/Total Data Compressed_KNN_model.joblib'),
    'total_data_compressed_lightgbm': load('/app/models/Total Data Compressed_LightGBM_model.joblib'),
    'total_data_compressed_linear_regression': load('/app/models/Total Data Compressed_LinearRegression_model.joblib'),
    'total_data_compressed_mlp': load('/app/models/Total Data Compressed_MLP_model.joblib'),
    'total_data_compressed_random_forest': load('/app/models/Total Data Compressed_RandomForest_model.joblib'),
    'total_data_compressed_svr': load('/app/models/Total Data Compressed_SVR_model.joblib'),
    'total_data_compressed_xgboost': load('/app/models/Total Data Compressed_XGBoost_model.joblib'),

    'total_data_filtered_catboost': load('/app/models/Total Data Filtered_CatBoost_model.joblib'),
    'total_data_filtered_knn': load('/app/models/Total Data Filtered_KNN_model.joblib'),
    'total_data_filtered_lightgbm': load('/app/models/Total Data Filtered_LightGBM_model.joblib'),
    'total_data_filtered_linear_regression': load('/app/models/Total Data Filtered_LinearRegression_model.joblib'),
    'total_data_filtered_mlp': load('/app/models/Total Data Filtered_MLP_model.joblib'),
    'total_data_filtered_random_forest': load('/app/models/Total Data Filtered_RandomForest_model.joblib'),
    'total_data_filtered_svr': load('/app/models/Total Data Filtered_SVR_model.joblib'),
    'total_data_filtered_xgboost': load('/app/models/Total Data Filtered_XGBoost_model.joblib'),

    'total_data_received_catboost': load('/app/models/Total Data Received_CatBoost_model.joblib'),
    'total_data_received_knn': load('/app/models/Total Data Received_KNN_model.joblib'),
    'total_data_received_lightgbm': load('/app/models/Total Data Received_LightGBM_model.joblib'),
    'total_data_received_linear_regression': load('/app/models/Total Data Received_LinearRegression_model.joblib'),
    'total_data_received_mlp': load('/app/models/Total Data Received_MLP_model.joblib'),
    'total_data_received_random_forest': load('/app/models/Total Data Received_RandomForest_model.joblib'),
    'total_data_received_svr': load('/app/models/Total Data Received_SVR_model.joblib'),
    'total_data_received_xgboost': load('/app/models/Total Data Received_XGBoost_model.joblib'),
}

@app.route('/predict/', methods=['POST'])
def predict_next_value():
    data = request.get_json()
    metric = data.get('metric')
    current_value = data.get('current_value')
    num_predictions = data.get('num_predictions')

    # Validação dos dados
    if not metric or not isinstance(metric, str):
        return jsonify({"error": "Invalid or missing 'metric'"}), 400
    if metric not in models:
        return jsonify({"error": "Metric not found"}), 404
    if current_value is None or not isinstance(current_value, (int, float)):
        return jsonify({"error": "Invalid or missing 'current_value'"}), 400
    if num_predictions is None or not isinstance(num_predictions, int) or num_predictions <= 0:
        return jsonify({"error": "Invalid or missing 'num_predictions'"}), 400

    model = models[metric]

    # Simulação de features ausentes com zeros
    # Supondo que o modelo espera N features, onde apenas uma é o valor atual
    num_features = model.feature_importances_.shape[0]  # Obtém o número de features esperadas pelo modelo
    input_data = np.zeros((1, num_features))
    input_data[0, 0] = current_value  # Assumindo que a primeira feature é o valor atual

    predictions = []

    for _ in range(num_predictions):
        next_value = model.predict(input_data.copy())[0]  # Copiar input_data para evitar read-only array
        predictions.append(next_value)
        input_data[0, 0] = next_value  # Atualiza o valor atual para a próxima previsão

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
