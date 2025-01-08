import time
import cv2
import numpy as np
import pyautogui

def calibrate(
    cap, eye_detector, calibration_points, capture_time=3, screen_width=None, screen_height=None, output_dir="./calibration_images"
):
    """
    Realiza la calibración y genera imágenes que muestran los bounding boxes de los ojos
    y los landmarks seleccionados.

    Args:
        cap: Objeto cv2.VideoCapture.
        eye_detector: Instancia del detector de ojos.
        calibration_points: Lista de puntos de calibración (coordenadas de pantalla).
        capture_time: Tiempo en segundos para cada punto de calibración.
        screen_width: Ancho de la pantalla.
        screen_height: Alto de la pantalla.
        output_dir: Directorio donde se guardarán las imágenes generadas.

    Returns:
        training_data: Lista de diccionarios con características de ojos y puntos de pantalla.
    """
    import os

    if screen_width is None or screen_height is None:
        screen_width, screen_height = pyautogui.size()

    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_data = []
    cv2.setWindowProperty("Eye Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for idx, (sx, sy) in enumerate(calibration_points):
        print(f"[CALIBRACIÓN] Mira al punto ({sx}, {sy}) durante {capture_time} s.")
        start_time = time.time()
        last_frame = None  # Último frame para este punto de calibración

        while (time.time() - start_time) < capture_time:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] No se pudo capturar el frame.")
                break

            last_frame = frame  # Actualizar el último frame capturado

            # Crear un canvas negro
            canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            # Dibujar el punto de calibración
            cv2.circle(canvas, (sx, sy), 20, (0, 0, 255), -1)

            # Mostrar el canvas
            cv2.imshow("Eye Tracking", canvas)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("[INFO] Calibración interrumpida por el usuario.")
                return training_data

        # Procesar el último frame
        if last_frame is not None:
            eye_features = eye_detector.get_eye_features(last_frame)
            if eye_features is not None:
                right_eye = eye_features["right_eye"]
                left_eye = eye_features["left_eye"]

                # Guardar características de calibración
                training_data.append({
                    "right_eye": right_eye,
                    "left_eye": left_eye,
                    "screen_position": (sx, sy)
                })

                # Generar visualización
                annotated_frame = last_frame.copy()
                h, w, _ = annotated_frame.shape

                # Dibujar bounding boxes y landmarks
                for bbox, landmarks, color, label in [
                    ([285, 261], right_eye, (255, 0, 0), "Right Eye"),  # Azul
                    ([46, 233], left_eye, (0, 255, 0), "Left Eye")     # Verde
                ]:
                    # Bounding box
                    x_min = int(eye_detector.face_mesh.process(last_frame).multi_face_landmarks[0].landmark[bbox[0]].x * w)
                    y_min = int(eye_detector.face_mesh.process(last_frame).multi_face_landmarks[0].landmark[bbox[0]].y * h)
                    x_max = int(eye_detector.face_mesh.process(last_frame).multi_face_landmarks[0].landmark[bbox[1]].x * w)
                    y_max = int(eye_detector.face_mesh.process(last_frame).multi_face_landmarks[0].landmark[bbox[1]].y * h)

                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 1)

                    # Dibujar landmarks
                    for (lx, ly) in landmarks:
                        print(lx, ly)
                        cv2.circle(annotated_frame, (int(lx), int(ly)), 1, color, -1)

                    # Etiqueta del bounding box
                    cv2.putText(
                        annotated_frame, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

                # Guardar la imagen
                output_path = os.path.join(output_dir, f"calibration_point_{idx + 1}.png")
                cv2.imwrite(output_path, annotated_frame)
                print(f"[INFO] Imagen de calibración guardada: {output_path}")
            else:
                print(f"[WARNING] No se detectaron características en el punto ({sx}, {sy}).")

    print(f"[INFO] Calibración completada. Total de muestras recogidas: {len(training_data)}.")
    return training_data
