import cv2
import pyautogui
from numpy import array
import time

from detection import EyeDetector
from calibration import calibrate
from model import train_model, predict_gaze
from cursor_functions import scroll_based_on_cursor_position

# Desactivar la función de seguridad de PyAutoGUI para que no se bloquee al mover el cursor a los bordes de la pantalla.
pyautogui.FAILSAFE = False


def smooth_position(current_x, current_y, new_x, new_y, alpha=0.95):
    """
    Smooth the position using exponential moving average.
    
    :param current_x: Current smoothed x position
    :param current_y: Current smoothed y position
    :param new_x: New x position
    :param new_y: New y position
    :param alpha: Smoothing factor
    :return: Smoothed x and y positions
    """
    if current_x is None or current_y is None:
        return new_x, new_y
    smoothed_x = alpha * current_x + (1 - alpha) * new_x
    smoothed_y = alpha * current_y + (1 - alpha) * new_y
    return smoothed_x, smoothed_y

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

    # 4. Definimos la resolución de pantalla
    screen_width, screen_height = pyautogui.size()

    # 5. Definir puntos de calibración en coordenadas de PANTALLA
    x_positions = [
        0,
        screen_width // 4,
        screen_width // 2,
        3 * screen_width // 4,
        screen_width - 1
    ]
    
    calibration_points = []

    # Linea SUPERIOR
    for x in x_positions:
        calibration_points.append((x, 0))  # y = 0 para la LINEA SUPERIOR

    # Linea INTERMEDIA SUPERIOR
    for x in x_positions:
        calibration_points.append((x, screen_height // 4))  # y = y/4 para la LINEA INTERMEDIA SUPERIOR

    # Linea CENTRAL
    for x in x_positions:
        calibration_points.append((x, screen_height // 2))  # y = y/2 para la LINEA INTERMEDIA

    # Linea INTERMEDIA INFERIOR
    for x in x_positions:
        calibration_points.append((x, 3 * screen_height // 4))  # y = 3*y/4 para la LINEA INTERMEDIA INFERIOR

    # Linea INFERIOR
    for x in x_positions:
        calibration_points.append((x, screen_height - 1))  # y = y para la LINEA INFERIOR

    # 6. Llamamos a la fase de calibración
    print("[INFO] Iniciando calibración...")
    
    
    training_data = calibrate(
        cap, 
        eye_detector,
        calibration_points,
        capture_time=3,
        screen_width=screen_width,
        screen_height=screen_height
    )
    

    
    
    print(f"[INFO] Calibración completada. Muestras recogidas: {len(training_data)}")

    if len(training_data) < 10:
        print("[ERROR] No hay suficientes muestras para entrenar.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Cierra la ventana de calibracion
    cv2.destroyWindow("Eye Tracking")

    # 7. Entrenamos el modelo
    print("[INFO] Entrenando modelo...")
    model = train_model(training_data)
    print("[INFO] Modelo entrenado. Fase de inferencia...")

    # 8. Filtros de suavizado (opcional)
    smoothed_x = None
    smoothed_y = None
    alpha = 0.9  # Ajusta entre 0.7 y 0.95 para más/menos suavidad

    print("[INFO] Moviendo el cursor con PyAutoGUI. Presiona ESC para salir.")

    last_scroll_time = time.time()
    scroll_delay = 4  # segundos

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mostrar la imagen de la cámara en una ventana
        cv2.imshow("Webcam Feed", frame)

        # Obtenemos características del ojo
        eye_feat = eye_detector.get_eye_features(frame)
        if eye_feat is not None:
            pred = predict_gaze(model, eye_feat)
            if pred:
                px, py = pred
                # Suavizamos un poco la posición
                smoothed_x, smoothed_y = smooth_position(smoothed_x, smoothed_y, px, py, alpha)

                # Movemos el cursor a la posición (suavizada) en la pantalla
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

                # Scroll based on cursor position
                last_scroll_time = scroll_based_on_cursor_position(smoothed_x, smoothed_y, screen_width, screen_height, scroll_delay, last_scroll_time)

                # Comprobamos si se ha detectado un guiño
                if eye_detector.detect_wink(frame) == 2:
                    break

        # Para terminar, usa ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
