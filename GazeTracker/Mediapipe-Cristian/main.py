# main.py
import cv2
import numpy as np
from detection import EyeDetector
from calibration import calibrate
from model import train_model, predict_gaze
from filterpy.kalman import KalmanFilter
from collections import deque
import pyautogui


# --- VARIABLES GLOBALES PARA SUAVIZADO ---
smoothed_x = None
smoothed_y = None
ALPHA = 0.9  # factor de suavizado




def smooth_prediction(px, py):
    """
    #Aplica un suavizado exponencial simple para reducir temblores.
    """
    global smoothed_x, smoothed_y, ALPHA
    if smoothed_x is None or smoothed_y is None:
        smoothed_x = px
        smoothed_y = py
    else:
        smoothed_x = ALPHA * smoothed_x + (1 - ALPHA) * px
        smoothed_y = ALPHA * smoothed_y + (1 - ALPHA) * py
    return int(smoothed_x), int(smoothed_y)


# Historial de puntos (tamaño limitado)
history_size = 15  # Número de muestras para suavizado
history_x = deque(maxlen=history_size)
history_y = deque(maxlen=history_size)

def smooth_prediction_mobile_mean(px, py):
    """
    Aplica un suavizado avanzado basado en una media móvil ponderada.
    """
    global history_x, history_y
    
    # Agregar los nuevos puntos al historial
    history_x.append(px)
    history_y.append(py)

    # Asignar pesos a las muestras (mayor peso a las más recientes)
    weights = [i + 1 for i in range(len(history_x))]  # [1, 2, 3, ..., history_size]
    weight_sum = sum(weights)

    # Calcular la media ponderada para x y y
    smoothed_x = sum(w * x for w, x in zip(weights, history_x)) / weight_sum
    smoothed_y = sum(w * y for w, y in zip(weights, history_y)) / weight_sum

    return int(smoothed_x), int(smoothed_y)


def smooth_prediction_kalman(px, py):
    """
    Aplica un suavizado a la predicción de la mirada utilizando un filtro de Kalman.

    Args:
      px: Coordenada x del punto predicho.
      py: Coordenada y del punto predicho.

    Returns:
      Una tupla con las coordenadas (x, y) suavizadas del punto.
    """
    global kf

    # Inicializar el filtro de Kalman si es necesario
    if not 'kf' in globals():
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([px, py, 0, 0])  # Estado inicial (posición y velocidad)
        kf.F = np.array([[1, 0, 1, 0],  # Matriz de transición de estado
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Matriz de observación
                         [0, 1, 0, 0]])
        kf.P *= 500  # Matriz de covarianza del error de estimación
        kf.R = 2  # Matriz de covarianza del ruido de medición

    # Obtener la predicción del filtro
    kf.predict()

    # Actualizar el filtro con la nueva medición
    kf.update(np.array([px, py]))

    # Devolver la posición estimada
    return int(kf.x[0]), int(kf.x[1])


def main():

    # 1. Inicializa la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    # 2. Crea la ventana "Eye Tracking" en modo fullscreen
    cv2.namedWindow("Eye Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 3. Instancia el detector de ojos
    eye_detector = EyeDetector(
        static_mode=False,
        max_faces=1,
        detection_confidence=0.7,
        tracking_confidence=0.7
    )

    # 4. Definimos la resolución de pantalla

    screen_width, screen_height = pyautogui.size()

    # 5. Definir puntos de calibración en coordenadas de PANTALLA
    #   Usamos puntos RELATIVOS al tamaño de la PANTALLA.

    x_positions = [
    0,
    screen_width // 4,
    screen_width // 2,
    3 * screen_width // 4,
    screen_width - 1
    ]
    
    # Lista para los puntos de Calibración dividiendo la pantalla horizontalmente en 3.
    calibration_points = []

    # Linea SUPERIOR
    for x in x_positions:
        calibration_points.append((x, 0))  # y = 0 para la LINEA SUPERIOR

    # Linea CENTRAL
    for x in x_positions:
        calibration_points.append((x, screen_height // 2))  # y = y/2 para la LINEA INTERMEDIA

    # Linea INFERIOR
    for x in x_positions:
        calibration_points.append((x, screen_height - 1))  # y = y para la LINEA INFERIOR


    # 6. Calibración (usa la MISMA ventana, en PANTALLA COMPLETA)
    training_data = calibrate(
        cap=cap,
        eye_detector=eye_detector,
        calibration_points=calibration_points,
        capture_time=3,
        screen_width=screen_width,
        screen_height=screen_height
    )
    print(f"[INFO] Calibración completada. Muestras totales: {len(training_data)}")

    if len(training_data) == 0:
        print("[ERROR] No se recogieron datos de calibración.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # 7. Entrenar modelo
    model = train_model(training_data)
    print("[INFO] Modelo entrenado.")

    # 8. Fase de Detección en la MISMA ventana
    print("[INFO] Iniciando inferencia (presiona ESC para salir).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # a) Creamos un canvas negro de la misma resolución de pantalla
        canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # b) Detectamos la mirada
        eye_feat = eye_detector.get_eye_features(frame)
        if eye_feat is not None:
            prediction = predict_gaze(model, eye_feat)
            if prediction:
                pred_x, pred_y = prediction
                # c) Suavizamos
                s_x, s_y = smooth_prediction_mobile_mean(pred_x, pred_y)
                # d) Dibujamos un círculo rojo donde se estima la mirada
                cv2.circle(canvas, (s_x, s_y), 20, (0, 0, 255), 2)

        # e) Mostramos en la ventana existente
        cv2.imshow("Eye Tracking", canvas)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
