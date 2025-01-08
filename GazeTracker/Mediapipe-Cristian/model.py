import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from catboost import CatBoostRegressor, Pool

def flatten_eye_features(eye_features, expected_size):
    """
    Flatten the array of eye landmarks into a single feature vector.
    Pads or truncates to ensure consistent size.
    """
    flat_features = eye_features.flatten()

    # Pad or truncate to ensure consistent size
    if len(flat_features) < expected_size:
        padding = expected_size - len(flat_features)
        flat_features = np.pad(flat_features, (0, padding), mode='constant')
    elif len(flat_features) > expected_size:
        flat_features = flat_features[:expected_size]

    return flat_features

def prepare_features(training_data):
    """
    Prepares the feature matrix (X) and target matrix (y) for training.
    """
    X = []
    y = []

    # Determine expected sizes based on the first sample
    sample = training_data[0]
    right_eye_size = sample["right_eye"].size
    left_eye_size = sample["left_eye"].size

    for sample in training_data:
        right_eye = flatten_eye_features(sample["right_eye"], expected_size=right_eye_size)
        left_eye = flatten_eye_features(sample["left_eye"], expected_size=left_eye_size)
        eye_features = np.concatenate([right_eye, left_eye])

        X.append(eye_features)
        y.append(sample["screen_position"])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), (right_eye_size, left_eye_size)

def grid_search_model(X, y):
    """
    Perform GridSearchCV to find the best hyperparameters for GradientBoostingRegressor.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0]
    }

    gbr = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    
    print("[INFO] Best parameters found: ", grid_search.best_params_)
    print("[INFO] Best score: ", -grid_search.best_score_)
    
    return grid_search.best_estimator_

def train_model_with_catboost(training_data):
    """
    Train CatBoostRegressor models for X and Y coordinates using GPU acceleration.
    """
    # Preparar datos
    X, y, expected_sizes = prepare_features(training_data)

    # Configuración de hiperparámetros para CatBoost
    cat_params = {
        'iterations': 2000,        # Número de iteraciones (árboles)
        'learning_rate': 0.0095,    # Tasa de aprendizaje
        'depth': 11,               # Profundidad máxima de los árboles
        'loss_function': 'RMSE',  # Función objetivo (Root Mean Squared Error)
        'task_type': 'GPU',       # Aceleración por GPU
        'random_seed': 42,        # Semilla para reproducibilidad
        'verbose': True          # Desactiva mensajes detallados de entrenamiento
    }

    # Entrenar modelos para X e Y
    print("[INFO] Entrenando modelo X con XGBoost...")
    model_x = CatBoostRegressor(**cat_params)
    model_x.fit(X, y[:, 0])

    print("[INFO] Entrenando modelo Y con XGBoost...")
    model_y = CatBoostRegressor(**cat_params)
    model_y.fit(X, y[:, 1])

    return (model_x, model_y), expected_sizes



def train_model_with_gridsearch(training_data):
    """
    Train Gradient Boosting Regressor models for X and Y coordinates using GridSearchCV.
    """
    X, y, expected_sizes = prepare_features(training_data)

    # Optimize and train the model for X coordinate
    print("[INFO] Optimizing model for X coordinate...")
    model_x = grid_search_model(X, y[:, 0])

    # Optimize and train the model for Y coordinate
    print("[INFO] Optimizing model for Y coordinate...")
    model_y = grid_search_model(X, y[:, 1])

    return (model_x, model_y), expected_sizes

def cross_validate_model(X, y, model):
    """
    Perform cross-validation on the given model.
    """
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = (-scores) ** 0.5
    print(f"[INFO] Cross-validated RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    return rmse_scores

def predict_gaze(models, eye_features, expected_sizes):
    """
    Predict the screen position (gaze point) based on eye landmarks using the trained models.
    """
    if eye_features is None:
        return None

    right_eye_size, left_eye_size = expected_sizes

    # Flatten and concatenate right and left eye features
    right_eye = flatten_eye_features(eye_features["right_eye"], expected_size=right_eye_size)
    left_eye = flatten_eye_features(eye_features["left_eye"], expected_size=left_eye_size)
    input_features = np.concatenate([right_eye, left_eye])

    # Predict using both models
    model_x, model_y = models
    pred_x = model_x.predict([input_features])[0]
    pred_y = model_y.predict([input_features])[0]

    return (int(pred_x), int(pred_y))
