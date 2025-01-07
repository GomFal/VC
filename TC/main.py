import cv2
import pyautogui
import numpy as np
import time

from detection import EyeDetector
from calibration import calibrate
from model import train_model, predict_gaze

def main():
    # 1. Abrimos la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    # 2. Creamos la ventana "Eye Tracking" para calibración e inferencia
    cv2.namedWindow("Eye Tracking", cv2.WINDOW_NORMAL)

    # 3. Creamos el detector de ojos
    eye_detector = EyeDetector(
        static_mode=False,
        max_faces=1,
        detection_confidence=0.7,
        tracking_confidence=0.7
    )

    # 4. Definimos la resolución real de tu pantalla (para calibración e inferencia)
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

    # 5. Definimos tus puntos de calibración (ejemplo: 6 puntos)
    calibration_points = [
        (100, 100),
        (1820, 100),
        (100, 980),
        (1820, 980),
        (960, 540),  # centro
        (1820, 540)  # extra
    ]
    capture_time = 3  # segundos por punto

    # 6. Llamamos a la fase de calibración
    print("[INFO] Iniciando calibración...")
    training_data = calibrate(
        cap, 
        eye_detector,
        calibration_points,
        capture_time=capture_time,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT
    )
    print(f"[INFO] Calibración completada. Muestras recogidas: {len(training_data)}")

    if len(training_data) < 10:
        print("[ERROR] No hay suficientes muestras para entrenar.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # 7. Entrenamos el modelo
    print("[INFO] Entrenando modelo...")
    model = train_model(training_data)
    print("[INFO] Modelo entrenado. Fase de inferencia...")

    # 8. Filtros de suavizado (opcional)
    smoothed_x = None
    smoothed_y = None
    alpha = 0.8  # Ajusta entre 0.7 y 0.95 para más/menos suavidad

    print("[INFO] Moviendo el cursor con PyAutoGUI. Presiona ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Obtenemos características del ojo
        eye_feat = eye_detector.get_eye_features(frame)
        if eye_feat is not None:
            pred = predict_gaze(model, eye_feat)
            if pred:
                px, py = pred
                # Suavizamos un poco la posición
                if smoothed_x is None:
                    smoothed_x, smoothed_y = px, py
                else:
                    smoothed_x = alpha * smoothed_x + (1 - alpha) * px
                    smoothed_y = alpha * smoothed_y + (1 - alpha) * py

                # Movemos el cursor a la posición (suavizada) en la pantalla
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

        # Para terminar, usa ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
