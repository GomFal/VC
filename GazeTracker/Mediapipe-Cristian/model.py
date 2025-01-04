# model.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


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
