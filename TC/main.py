# main.py
import cv2
import numpy as np
from detection import EyeDetector
from calibration import calibrate
from model import train_model, predict_gaze

# --- VARIABLES GLOBALES PARA SUAVIZADO ---
smoothed_x = None
smoothed_y = None
ALPHA = 0.8  # factor de suavizado

def smooth_prediction(px, py):
    """
    Aplica un suavizado exponencial simple para reducir temblores.
    """
    global smoothed_x, smoothed_y, ALPHA
    if smoothed_x is None or smoothed_y is None:
        smoothed_x = px
        smoothed_y = py
    else:
        smoothed_x = ALPHA * smoothed_x + (1 - ALPHA) * px
        smoothed_y = ALPHA * smoothed_y + (1 - ALPHA) * py
    return int(smoothed_x), int(smoothed_y)


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
    screen_width = 1920
    screen_height = 1080

    # 5. Definir puntos de calibración en coordenadas de PANTALLA
    #    (ajusta si tu pantalla real tiene otra resolución)
    calibration_points = [
        (100, 100),
        (1820, 100),
        (100, 980),
        (1820, 980),
        (960, 540),   # centro
        (1820, 540)   # uno extra
    ]

    # 6. Calibración (usa la MISMA ventana, en fullscreen)
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
                s_x, s_y = smooth_prediction(pred_x, pred_y)
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
