# detection.py
import cv2
import mediapipe as mp
import numpy as np


# Inicializamos la solución de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

class EyeDetector:
    """
    Clase para encapsular la detección de ojos / iris usando MediaPipe Face Mesh.
    """
    def __init__(self, static_mode=False, max_faces=1, detection_confidence=0.5, tracking_confidence=0.5):
        # Refinar landmarks = True (importante para que reconozca el iris)
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def get_eye_features(self, frame):
        """
        Dado un frame BGR, retorna (eye_features) que podrían ser:
            - [x_iris_izq, y_iris_izq, x_iris_der, y_iris_der]
            o alguna otra representación.

        Devuelve None si no se detectó rostro.
        """
        # 1. Convertir BGR a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Procesar con Face Mesh
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None  # No se detectó ningún rostro

        # Asumimos solo 1 rostro, tomamos el primero
        face_landmarks = results.multi_face_landmarks[0]

        # Indices de landmarks relevantes para el iris.
        # Hay diferentes mapas: 
        #   iris right (474, 475, 476, 477) 
        #   iris left (469, 470, 471, 472)
        # Tomaremos uno de los landmarks centrales para cada iris.
        # (ejemplo: 468 = centro aproximado del iris izquierdo, 473 = centro aproximado del iris derecho
        #  según la doc de MediaPipe, pero a veces varía. Revisar la lista completa de landmarks).

        # Para este ejemplo, usemos:
        IRIS_LEFT_CENTER = 468
        IRIS_RIGHT_CENTER = 473

        h, w, _ = frame.shape

        # Accedemos a los landmarks
        iris_left_lm = face_landmarks.landmark[IRIS_LEFT_CENTER]
        iris_right_lm = face_landmarks.landmark[IRIS_RIGHT_CENTER]

        # Convertir coordenadas normalizadas [0,1] → pixeles
        x_left = int(iris_left_lm.x * w)
        y_left = int(iris_left_lm.y * h)
        x_right = int(iris_right_lm.x * w)
        y_right = int(iris_right_lm.y * h)

        # Devolvemos como array np [x_izq, y_izq, x_der, y_der]
        return np.array([x_left, y_left, x_right, y_right], dtype=np.float32)

