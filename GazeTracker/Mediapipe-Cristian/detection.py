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
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def get_eye_features(self, frame):
        """
        Dado un frame BGR, retorna las coordenadas relativas al rectángulo 
        definido por los landmarks 285, 261 para el ojo derecho y 46, 233 
        para el izquierdo, de los landmarks 471, 468, 469 en el ojo izquierdo 
        y 476, 473, 474 en el derecho.

        Devuelve un diccionario con las características de los ojos:
            {
                "right_eye": [[x1, y1], [x2, y2], ...],
                "left_eye": [[x1, y1], [x2, y2], ...]
            }

        Devuelve None si no se detectó rostro.
        """
        # Convertir BGR a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con Face Mesh
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None  # No se detectó ningún rostro

        # Asumimos solo 1 rostro, tomamos el primero
        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        # Puntos que definen los rectángulos de los ojos
        RIGHT_EYE_RECT = [285, 261]  # Superior izquierdo, inferior derecho
        LEFT_EYE_RECT = [46, 233]

        # Landmarks de los ojos
        LEFT_EYE_LANDMARKS = [471, 468, 469]
        RIGHT_EYE_LANDMARKS = [476, 473, 474]

        # Función para obtener landmarks dentro de un rectángulo
        def get_landmarks_in_bbox(bbox_points, eye_landmarks_indices, landmarks):
            x_min = int(landmarks.landmark[bbox_points[0]].x * w)
            y_min = int(landmarks.landmark[bbox_points[0]].y * h)
            x_max = int(landmarks.landmark[bbox_points[1]].x * w)
            y_max = int(landmarks.landmark[bbox_points[1]].y * h)

            # Verificar área válida
            if x_max == x_min or y_max == y_min:
                return []  # Devolver lista vacía para mantener consistencia en caso de que el rectángulo sea 0

            # Obtener coordenadas relativas al rectángulo
            eye_landmarks_relative = []
            eye_landmarks = []
            for i in eye_landmarks_indices:
                x = int(landmarks.landmark[i].x * w)
                y = int(landmarks.landmark[i].y * h)
                x_relative = (x - x_min) / (x_max - x_min)  # Normalizado entre 0 y 1
                y_relative = (y - y_min) / (y_max - y_min)  # Normalizado entre 0 y 1
                eye_landmarks_relative.append([x_relative, y_relative])
                eye_landmarks.append([x, y])                # Vemos cual es más accurate de los dos, si las coordenadas relativas o las absolutas

            return eye_landmarks

        # Obtener landmarks para cada ojo
        right_eye_landmarks = get_landmarks_in_bbox(RIGHT_EYE_RECT, RIGHT_EYE_LANDMARKS, face_landmarks)
        left_eye_landmarks = get_landmarks_in_bbox(LEFT_EYE_RECT, LEFT_EYE_LANDMARKS, face_landmarks)

        if not right_eye_landmarks or not left_eye_landmarks:
            print("[INFO] Frame omitido por rectángulo inválido.")
            return None

        # Retornar las características de los ojos
        return {
            "right_eye": np.array(right_eye_landmarks, dtype=np.float32),
            "left_eye": np.array(left_eye_landmarks, dtype=np.float32)
        }