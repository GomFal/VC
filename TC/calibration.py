import time
import cv2
import numpy as np
import pyautogui
import os


def calibrate(cap, eye_detector, calibration_points, capture_time=3, screen_width=None, screen_height=None):
    """
    Realiza la calibración usando UNA SOLA ventana, que se asume ya creada en main.py.
    Crea un lienzo (canvas) negro del tamaño de la pantalla y dibuja puntos de calibración en ROJO.
    Mientras tanto, captura frames de la webcam en segundo plano para registrar eye_features.
    
    Returns:
        training_data: lista de (eye_features, xPant, yPant)
    """
    training_data = []
    image_index = 0
    cv2.setWindowProperty("Eye Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if screen_width is None or screen_height is None:
        screen_width, screen_height = pyautogui.size()

    for (sx, sy) in calibration_points:
        print(f"[CALIBRACIÓN] Mira al punto ({sx}, {sy}) durante {capture_time} s.")
        start_time = time.time()
        frames = []

        while (time.time() - start_time) < capture_time:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1) Creamos un canvas negro del tamaño de pantalla
            canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            
            # 2) Calculamos el tamaño del círculo
            elapsed_time = time.time() - start_time
            circle_radius = int(30 * (1 - elapsed_time / capture_time)) + 1
            
            # 3) Dibujamos el punto de calibración en ROJO -> BGR: (0,0,255)
            cv2.circle(canvas, (sx, sy), circle_radius, (0, 0, 255), -1)

            # 4) Guardamos el frame para procesarlo después
            frames.append(frame)

            # 5) Mostramos en la MISMA ventana ("Eye Tracking")
            cv2.imshow("Eye Tracking", canvas)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        # Procesamos el último frame capturado para extraer las características del ojo
        if frames:
            eye_feat = eye_detector.get_eye_features(frames[-1])
            if eye_feat is not None:
                # Asociamos la posición real de pantalla (sx, sy)
                training_data.append((eye_feat, sx, sy))
                print(training_data[-1])

    print(training_data)
    return training_data
