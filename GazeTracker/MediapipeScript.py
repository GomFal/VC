import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

mp_face_mesh = mp.solutions.face_mesh

# Inicializar la detección de malla facial
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Índices de los puntos de referencia para los ojos
LEFT_EYE_INDEXES = [33, 133]
RIGHT_EYE_INDEXES = [362, 263]

# Función para calcular la dirección de la mirada
def get_gaze_direction(image_points, image_width, image_height):
    # Calcular la posición media de los ojos
    left_eye_center = np.mean(image_points[LEFT_EYE_INDEXES], axis=0)
    right_eye_center = np.mean(image_points[RIGHT_EYE_INDEXES], axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2

    # Normalizar las coordenadas de los ojos
    normalized_eye_center = eye_center / [image_width, image_height]

    # Calcular la dirección de la mirada
    gaze_direction = normalized_eye_center - [0.5, 0.5]

    return gaze_direction

def calibrar_mirada(cap, face_mesh, image_width, image_height):
  """
  Calibra la mirada del usuario utilizando 9 puntos en la pantalla.

  Args:
    cap: Objeto cv2.VideoCapture para capturar video.
    face_mesh: Objeto Mediapipe FaceMesh para la detección de malla facial.
    image_width: Ancho de la imagen.
    image_height: Alto de la imagen.

  Returns:
    Una tupla con dos matrices NumPy:
      - `gaze_directions`: Matriz de 9x2 con las direcciones de la mirada 
                            para cada punto de calibración.
      - `screen_points`: Matriz de 9x2 con las coordenadas de los puntos 
                         de calibración en la pantalla.
  """

  # Coordenadas de los 9 puntos de calibración en la pantalla
  screen_points = np.array([
      [image_width * 0.2, image_height * 0.2],
      [image_width * 0.5, image_height * 0.2],
      [image_width * 0.8, image_height * 0.2],
      [image_width * 0.2, image_height * 0.5],
      [image_width * 0.5, image_height * 0.5],
      [image_width * 0.8, image_height * 0.5],
      [image_width * 0.2, image_height * 0.8],
      [image_width * 0.5, image_height * 0.8],
      [image_width * 0.8, image_height * 0.8],
  ])

  gaze_directions = []

  for i, point in enumerate(screen_points):
    # Mostrar el punto de calibración en la pantalla
    success, image = cap.read()
    cv2.circle(image, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)
    cv2.imshow('MediaPipe Face Mesh', image)

    # Esperar a que el usuario presione una tecla
    cv2.waitKey(0)

    # Capturar la dirección de la mirada
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        image_points = np.array([
            (int(landmark.x * image_width), int(landmark.y * image_height))
            for landmark in face_landmarks.landmark
        ])
        gaze_direction = get_gaze_direction(image_points, image_width, image_height)
        gaze_directions.append(gaze_direction)

  return np.array(gaze_directions), screen_points

def interpolar_mirada(gaze_direction, gaze_directions, screen_points):
  """
  Interpola la posición de la mirada en la pantalla a partir de la 
  dirección de la mirada y los datos de calibración.

  Args:
    gaze_direction: Dirección de la mirada actual.
    gaze_directions: Matriz de direcciones de la mirada de calibración.
    screen_points: Matriz de coordenadas de los puntos de calibración.

  Returns:
    Las coordenadas (x, y) del punto en la pantalla donde se estima que 
    el usuario está mirando.
  """

  # Crear una función de interpolación 2D con 3 puntos en cada dimensión
  x = np.linspace(np.min(gaze_directions[:, 0]), np.max(gaze_directions[:, 0]), 3)
  y = np.linspace(np.min(gaze_directions[:, 1]), np.max(gaze_directions[:, 1]), 3)
  f = RegularGridInterpolator((x, y), screen_points.reshape(3, 3, 2), method='linear', bounds_error=False, fill_value=None)

  # Interpolar la posición de la mirada
  gaze_point = f(np.array([gaze_direction[0], gaze_direction[1]]))

  return gaze_point

# Capturar video desde la cámara web
cap = cv2.VideoCapture(0)

# Obtener el ancho y alto de la imagen
success, image = cap.read()
if not success:
  print("Ignoring empty camera frame.")
  # If loading a video, use 'break' instead of 'continue'.
  

image_height, image_width, _ = image.shape

# Calibrar la mirada
gaze_directions, screen_points = calibrar_mirada(cap, face_mesh, image_width, image_height)

while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Convertir la imagen a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Dibujar la malla facial en la imagen
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Obtener las coordenadas de los puntos de referencia faciales
            image_points = np.array([
                (int(landmark.x * image_width), int(landmark.y * image_height))
                for landmark in face_landmarks.landmark
            ])

            # Calcular la dirección de la mirada
            gaze_direction = get_gaze_direction(image_points, image_width, image_height)

            # Interpolar la posición de la mirada en la pantalla
            gaze_point = interpolar_mirada(gaze_direction, gaze_directions, screen_points)

            # Dibujar un punto en el lugar donde la persona está mirando
            cv2.circle(image, (int(gaze_point[0]), int(gaze_point[1])), 5, (255, 0, 0), -1)

    # Mostrar la imagen
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()