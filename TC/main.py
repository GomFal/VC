import cv2
import pyautogui
import numpy as np
from numpy import array

# Desactivar la función de seguridad de PyAutoGUI para que no se bloquee al mover el cursor a los bordes de la pantalla


import numpy as np
import time

from detection import EyeDetector
from calibration import calibrate
from model import train_model, predict_gaze
from cursor_functions import scroll_based_on_cursor_position

import pygetwindow as gw


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
    
    #Datos de la última calibración por si queremos hacer pruebas sin calibrar
    training_data = [(array([0.7924404 , 0.53470726, 0.75243786, 0.52672839, 0.83203329,
       0.5426558 , 0.31402359, 0.50705016, 0.35692939, 0.50946436,
       0.26931364, 0.50360493]), 0, 0), (array([0.77004869, 0.55221787, 0.73124922, 0.54575996, 0.80882255,
       0.5577769 , 0.28614726, 0.53787872, 0.32991577, 0.54161798,
       0.24127422, 0.53382725]), 480, 0), (array([0.7486814 , 0.52711915, 0.70497162, 0.5232742 , 0.79246159,
       0.53005339, 0.26186022, 0.5169178 , 0.30586951, 0.5152595 ,
       0.21728369, 0.51794585]), 960, 0), (array([0.72231195, 0.52171073, 0.67417689, 0.51768267, 0.77122704,
       0.52513277, 0.24337555, 0.51425017, 0.28511213, 0.50889429,
       0.20108388, 0.51928779]), 1440, 0), (array([0.70503499, 0.51224834, 0.65755741, 0.51001555, 0.75375994,
       0.51285287, 0.2340794 , 0.49672261, 0.27360172, 0.48980612,
       0.19425481, 0.50339684]), 1919, 0), (array([0.76002221, 0.5444053 , 0.71738279, 0.53815392, 0.80236695,
       0.55014938, 0.27764524, 0.51983484, 0.32048512, 0.52176237,
       0.23392597, 0.51715571]), 480, 270), (array([0.74311287, 0.54787872, 0.69949742, 0.54113734, 0.78707925,
       0.55334236, 0.25449978, 0.54968308, 0.29845537, 0.5490895 ,
       0.20984466, 0.54997459]), 960, 270), (array([0.72599944, 0.52989733, 0.67825592, 0.5244858 , 0.7740679 ,
       0.53474002, 0.24910183, 0.54019879, 0.29187576, 0.5348559 ,
       0.20571327, 0.54522683]), 1440, 270), (array([0.78346332, 0.57609495, 0.74378328, 0.56818527, 0.8231485 ,
       0.58380766, 0.3094678 , 0.56700014, 0.35422546, 0.57041815,
       0.26394252, 0.56299813]), 0, 540), (array([0.76468662, 0.54724947, 0.72136831, 0.53847171, 0.80776684,
       0.55521556, 0.27777017, 0.54433219, 0.32363057, 0.54621765,
       0.23118216, 0.54181941]), 480, 540), (array([0.74315649, 0.54655118, 0.70229454, 0.54026293, 0.78480613,
       0.55210993, 0.24795209, 0.54045656, 0.29104649, 0.53901936,
       0.20420932, 0.54159129]), 960, 540), (array([0.72490467, 0.53969719, 0.68379125, 0.53338785, 0.76705148,
       0.54464233, 0.23654824, 0.54078975, 0.27852485, 0.53818065,
       0.19389095, 0.54326673]), 1440, 540), (array([0.71145341, 0.54122913, 0.66509969, 0.53620075, 0.75922466,
       0.54454176, 0.22946488, 0.52310625, 0.26893403, 0.5200696 ,
       0.18963572, 0.52660216]), 1919, 540), (array([0.77706542, 0.57864679, 0.73598338, 0.56856509, 0.81810584,
       0.58767043, 0.29665889, 0.55225658, 0.34206933, 0.55303312,
       0.25034408, 0.55025897]), 480, 810), (array([0.76031904, 0.5472463 , 0.71992584, 0.54234021, 0.80099742,
       0.55128801, 0.26804123, 0.54261479, 0.31031103, 0.54043253,
       0.22488774, 0.54416461]), 960, 810), (array([0.72927764, 0.55874184, 0.68374474, 0.55617285, 0.77584048,
       0.55982019, 0.24980999, 0.54657797, 0.29409026, 0.54532546,
       0.20484418, 0.54720046]), 1440, 810), (array([0.77860295, 0.58017503, 0.73489874, 0.57087455, 0.82177373,
       0.58883538, 0.30195471, 0.56090912, 0.34987688, 0.56372811,
       0.25345908, 0.55754978]), 0, 1079), (array([0.76475763, 0.55958742, 0.72071324, 0.55067854, 0.80828801,
       0.56724613, 0.27715642, 0.531366  , 0.32392935, 0.53236565,
       0.22954702, 0.52949706]), 480, 1079), (array([0.75386438, 0.55652358, 0.71216633, 0.55171039, 0.79578076,
       0.56035956, 0.26272756, 0.54594728, 0.30712376, 0.54385791,
       0.21719378, 0.54699306]), 960, 1079), (array([0.7365744 , 0.55912828, 0.68826421, 0.55527359, 0.78561644,
       0.56233895, 0.25036288, 0.54481254, 0.29189301, 0.5405646 ,
       0.20828944, 0.54810081]), 1440, 1079), (array([0.71829686, 0.54415112, 0.66924549, 0.54073336, 0.76827297,
       0.54591846, 0.24038522, 0.54805031, 0.28052702, 0.54495202,
       0.1997122 , 0.55071913]), 1919, 1079)]
    

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
