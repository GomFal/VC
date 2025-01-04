# calibration.py
import time
import cv2
import numpy as np

def calibrate(cap, eye_detector, calibration_points, capture_time=3, screen_width=1920, screen_height=1080):
    """
    Realiza la calibración usando UNA SOLA ventana, que se asume ya creada en main.py.
    Crea un lienzo (canvas) negro del tamaño de la pantalla y dibuja puntos de calibración en ROJO.
    Mientras tanto, captura frames de la webcam en segundo plano para registrar eye_features.
    
    Returns:
        training_data: lista de (eye_features, xPant, yPant)
    """
    training_data = []
    cv2.setWindowProperty("Eye Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for (sx, sy) in calibration_points:
        print(f"[CALIBRACIÓN] Mira al punto ({sx}, {sy}) durante {capture_time} s.")
        start_time = time.time()

        while (time.time() - start_time) < capture_time:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1) Creamos un canvas negro del tamaño de pantalla
            canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            
            # 2) Dibujamos el punto de calibración en ROJO -> BGR: (0,0,255)
            cv2.circle(canvas, (sx, sy), 10, (0, 0, 255), -1)

            # 3) Obtenemos las características del ojo (no lo mostramos)
            eye_feat = eye_detector.get_eye_features(frame)
            if eye_feat is not None:
                # Asociamos la posición real de pantalla (sx, sy)
                training_data.append((eye_feat, sx, sy))

            # 4) Mostramos en la MISMA ventana ("Eye Tracking")
            cv2.imshow("Eye Tracking", canvas)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    return training_data
