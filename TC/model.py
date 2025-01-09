# model.py
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Modelo de regresión lineal para predecir la posición de la mirada. Es simple pero funciona mejor que cualquier modelo que hayamos probado.

def train_model(training_data):
    X = []
    y = []
    for (eye_feat, sx, sy) in training_data:
        X.append(eye_feat)
        y.append([sx, sy])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_gaze(model, eye_feat):
    if eye_feat is None:
        return None
    pred = model.predict([eye_feat])[0]
    return (int(pred[0]), int(pred[1]))

# Modelo de Random Forest para predecir la posición de la mirada. No se usa ya que el modelo de regresión lineal funciona mejor.
def train_random_forest(training_data):
    X = []
    y = []
    for (eye_feat, sx, sy) in training_data:
        X.append(eye_feat)
        y.append([sx, sy])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation MSE: {mse}")
        
    return model

def predict_gaze_rf(model, eye_feat):
    if eye_feat is None:
        return None
    pred = model.predict(np.array([eye_feat], dtype=np.float32))[0]
    return (int(pred[0]), int(pred[1]))