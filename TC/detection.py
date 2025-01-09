import cv2
import mediapipe as mp
import numpy as np
import os
import time
import pyautogui

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
        self.wink_start_time = None

    def get_eye_features(self, frame):
        """
        Dado un frame BGR, retorna (eye_features) que son las coordenadas normalizadas de los landmarks:
            - [x_iris_izq_1, y_iris_izq_1, x_iris_izq_2, y_iris_izq_2, x_iris_izq_3, y_iris_izq_3, 
               x_iris_der_1, y_iris_der_1, x_iris_der_2, y_iris_der_2, x_iris_der_3, y_iris_der_3]

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
        LEFT_EYE_LANDMARKS = [473, 476, 474]
        RIGHT_EYE_LANDMARKS = [468, 469, 471]

        FACE_BOUNDING_BOX_LANDMARKS = [21, 447]

        h, w, _ = frame.shape

        # Obtener las coordenadas de los landmarks de la caja delimitadora
        bbox_coords = np.array([[face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] for idx in FACE_BOUNDING_BOX_LANDMARKS])
        bbox_x_min, bbox_y_min = np.min(bbox_coords, axis=0)
        bbox_x_max, bbox_y_max = np.max(bbox_coords, axis=0)

        # Accedemos a los landmarks de cada ojo y los normalizamos en el espacio de la caja delimitadora
        left_eye_coords = np.array([[(face_landmarks.landmark[idx].x * w - bbox_x_min) / (bbox_x_max - bbox_x_min), 
                                     (face_landmarks.landmark[idx].y * h - bbox_y_min) / (bbox_y_max - bbox_y_min)] for idx in LEFT_EYE_LANDMARKS])
        right_eye_coords = np.array([[(face_landmarks.landmark[idx].x * w - bbox_x_min) / (bbox_x_max - bbox_x_min), 
                                      (face_landmarks.landmark[idx].y * h - bbox_y_min) / (bbox_y_max - bbox_y_min)] for idx in RIGHT_EYE_LANDMARKS])

        # Denormalize the coordinates
        left_eye_coords_denorm = np.array([[coord[0] * (bbox_x_max - bbox_x_min) + bbox_x_min, coord[1] * (bbox_y_max - bbox_y_min) + bbox_y_min] for coord in left_eye_coords])
        right_eye_coords_denorm = np.array([[coord[0] * (bbox_x_max - bbox_x_min) + bbox_x_min, coord[1] * (bbox_y_max - bbox_y_min) + bbox_y_min] for coord in right_eye_coords])

        # Devolvemos las coordenadas normalizadas de los landmarks de cada ojo en el espacio de la caja delimitadora
        return np.concatenate((left_eye_coords.flatten(), right_eye_coords.flatten()), axis=0)

    def detect_wink(self, frame):
        """
        Detecta si el usuario ha guiñado un ojo basándose en los landmarks de los párpados superior e inferior.
        
        Args:
            frame: El frame en el que se realizará la detección.
        
        Returns:
            True si se detecta un guiño, False en caso contrario.
        """

        # Indices de landmarks relevantes para la caja delimitadora
        FACE_BOUNDING_BOX_LANDMARKS = [21, 447]

        # Obtener las características del ojo
        eye_features = self.get_eye_features(frame)
        if eye_features is None:
            self.wink_start_time = None
            return False  # No se detectó ningún rostro

        # Indices de landmarks relevantes para los párpados superior e inferior.
        LEFT_EYE_TOP_LANDMARK = 386
        LEFT_EYE_BOTTOM_LANDMARK = 374
        RIGHT_EYE_TOP_LANDMARK = 159
        RIGHT_EYE_BOTTOM_LANDMARK = 145

        # Convertir BGR a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar con Face Mesh
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            self.wink_start_time = None
            return False  # No se detectó ningún rostro

        # Asumimos solo 1 rostro, tomamos el primero
        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        # Obtener las coordenadas de los landmarks de los párpados
        left_eye_top = face_landmarks.landmark[LEFT_EYE_TOP_LANDMARK]
        left_eye_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM_LANDMARK]
        right_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP_LANDMARK]
        right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM_LANDMARK]

        # Obtener las coordenadas de los landmarks de la caja delimitadora
        bbox_coords = np.array([[face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] for idx in FACE_BOUNDING_BOX_LANDMARKS])
        bbox_x_min, bbox_y_min = np.min(bbox_coords, axis=0)
        bbox_x_max, bbox_y_max = np.max(bbox_coords, axis=0)

        # Calcular la distancia entre los párpados superior e inferior para cada ojo
        left_eye_distance = np.linalg.norm(
            np.array([left_eye_top.x * w, left_eye_top.y * h]) - np.array([left_eye_bottom.x * w, left_eye_bottom.y * h])
        )
        right_eye_distance = np.linalg.norm(
            np.array([right_eye_top.x * w, right_eye_top.y * h]) - np.array([right_eye_bottom.x * w, right_eye_bottom.y * h])
        )

        # Calcular la altura de la caja delimitadora
        bbox_height = bbox_y_max - bbox_y_min

        # Definir un umbral como un porcentaje de la altura de la caja delimitadora
        WINK_THRESHOLD_PERCENTAGE = 0.1  # Ajustar este valor según sea necesario
        wink_threshold = WINK_THRESHOLD_PERCENTAGE * bbox_height

        # Detectar si ambos ojos están cerrados si la distancia entre los párpados es menor que el umbral
        if left_eye_distance < wink_threshold and right_eye_distance < wink_threshold:
                if self.both_eyes_closed_start_time is None:
                    self.both_eyes_closed_start_time = time.time()
                elif time.time() - self.both_eyes_closed_start_time > 3:
                    print("Ambos ojos cerrados por más de 3 segundos. Se cierra el programa.")
                    return 2
        
        # Detectar guiño si un ojo está cerrado y el otro abierto
        elif (left_eye_distance < wink_threshold and right_eye_distance >= wink_threshold) or (left_eye_distance >= wink_threshold and right_eye_distance < wink_threshold):
            if self.wink_start_time is None:
                self.wink_start_time = time.time()
            elif time.time() - self.wink_start_time > 3:
                print("Guiño detectado por más de 3 segundos")
                pyautogui.hotkey('alt', 'f4')
                time.sleep(2)                   # Esperamos para que no se detecte otro guiño seguido
                return True
        else:
            self.wink_start_time = None
            self.both_eyes_closed_start_time = None

        return False
    
    def detect_both_eyes_closed(self, frame):
            """
            Detecta si el usuario ha cerrado ambos ojos por más de 3 segundos basándose en los landmarks de los párpados superior e inferior.
            
            Args:
                frame: El frame en el que se realizará la detección.
            
            Returns:
                True si se detecta que ambos ojos están cerrados por más de 3 segundos, False en caso contrario.
            """
            # Indices de landmarks relevantes para la caja delimitadora
            FACE_BOUNDING_BOX_LANDMARKS = [21, 447]

            # Obtener las características del ojo
            eye_features = self.get_eye_features(frame)
            if eye_features is None:
                self.both_eyes_closed_start_time = None
                return False  # No se detectó ningún rostro

            # Indices de landmarks relevantes para los párpados superior e inferior.
            LEFT_EYE_TOP_LANDMARK = 386
            LEFT_EYE_BOTTOM_LANDMARK = 374
            RIGHT_EYE_TOP_LANDMARK = 159
            RIGHT_EYE_BOTTOM_LANDMARK = 145

            # Convertir BGR a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar con Face Mesh
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                self.both_eyes_closed_start_time = None
                return False  # No se detectó ningún rostro

            # Asumimos solo 1 rostro, tomamos el primero
            face_landmarks = results.multi_face_landmarks[0]

            h, w, _ = frame.shape

            # Obtener las coordenadas de los landmarks de los párpados
            left_eye_top = face_landmarks.landmark[LEFT_EYE_TOP_LANDMARK]
            left_eye_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM_LANDMARK]
            right_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP_LANDMARK]
            right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM_LANDMARK]

            # Obtener las coordenadas de los landmarks de la caja delimitadora
            bbox_coords = np.array([[face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] for idx in FACE_BOUNDING_BOX_LANDMARKS])
            bbox_x_min, bbox_y_min = np.min(bbox_coords, axis=0)
            bbox_x_max, bbox_y_max = np.max(bbox_coords, axis=0)

            # Calcular la distancia entre los párpados superior e inferior para cada ojo
            left_eye_distance = np.linalg.norm(
                np.array([left_eye_top.x * w, left_eye_top.y * h]) - np.array([left_eye_bottom.x * w, left_eye_bottom.y * h])
            )
            right_eye_distance = np.linalg.norm(
                np.array([right_eye_top.x * w, right_eye_top.y * h]) - np.array([right_eye_bottom.x * w, right_eye_bottom.y * h])
            )

            # Calcular la altura de la caja delimitadora
            bbox_height = bbox_y_max - bbox_y_min

            # Definir un umbral como un porcentaje de la altura de la caja delimitadora
            EYES_CLOSED_THRESHOLD_PERCENTAGE = 0.15  # Ajustar este valor según sea necesario
            eyes_closed_threshold = EYES_CLOSED_THRESHOLD_PERCENTAGE * bbox_height

            # Detectar si ambos ojos están cerrados si la distancia entre los párpados es menor que el umbral
            if left_eye_distance < eyes_closed_threshold and right_eye_distance < eyes_closed_threshold:
                if self.both_eyes_closed_start_time is None:
                    self.both_eyes_closed_start_time = time.time()
                elif time.time() - self.both_eyes_closed_start_time > 3:
                    print("Ambos ojos cerrados por más de 3 segundos")
                    return True
            else:
                self.both_eyes_closed_start_time = None

            return False
