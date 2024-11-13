## Proyecto de Filtros Faciales: Máscara de Dragón con efecto de Fuego y Filtro de fiesta.

Este proyecto implementa dos filtros faciales. El primero es un filtro facial que coloca una máscara de dragón sobre la cara detectada en tiempo real. Además, cuando el sistema detecta que la boca está abierta, se generan emojis de fuego que simulan una respiración de fuego, siguiendo la posición de la boca y un ángulo ajustable. La detección facial y del estado de la boca se logra mediante el uso de la librería `dlib` para reconocimiento facial, y se visualizan en un flujo de video en tiempo real.
El segundo filtro, utiliza un detector de sonrisa para activar un efecto festivo. Cuando se detecta una sonrisa, el filtro superpone una corona brillante sobre la cabeza del usuario, ajustándose en tiempo real a sus movimientos para dar la impresión de que está realmente puesta. Además, confeti de colores que cae por toda la pantalla, generando una atmósfera de celebración. Al mismo tiempo, se activan conos de luz que simulan luces de discoteca.

## Autores
[![GitHub](https://img.shields.io/badge/GitHub-Javier%20Gómez%20Falcón-red?style=flat-square&logo=github)](https://github.com/GomFal)
[![GitHub](https://img.shields.io/badge/GitHub-Cristian%20Marrero%20Vega-blue?style=flat-square&logo=github)](https://github.com/XxMARRExX)

## Tecnologías
  - Python

## Librerías 
  - OpenCV
  - dlib
  - NumPy

## Modelo de detector Facial usado para la máscara del Dragón.
Ambos filtros utilizan el modelo preentrenado `shape_predictor_68_face_landmarks.dat` para la detección de 68 puntos faciales clave. Este modelo se emplea para detectar la posición de la boca y otros puntos de referencia para alinear la máscara de dragón y posicionar los efectos de fuego. En el caso del filtro festivo, se utiliza para poder tener una referencia sobre donde se coloca la corona y para poder detectar la sonrisa.

## Procedimiento seguido para la máscara del Dragón:
  **1. Inicialización de los Modelos y Recursos.**  
   Se carga el modelo de predicción de puntos faciales `shape_predictor_68_face_landmarks.dat` de dlib para el seguimiento de puntos faciales, la imagen de la máscara de dragón y el efecto de fuego en formato PNG, ambos con transparencia.

  **2. Detección de Rostros y Colocación de la Máscara.**  
   Para cada fotograma capturado de la cámara, se detectan rostros utilizando el detector frontal de rostros de `dlib`. Una vez identificado el rostro:
   - Se ajusta el tamaño de la máscara según el ancho de la cara y se alinea con los puntos faciales clave para que siga la inclinación de la cara.
   - Se superpone la máscara de dragón en el rostro detectado usando un canal alfa que permite manejar la transparencia.

  **3. Detección del Estado de la Boca.**  
   Para detectar si la boca está abierta:
   - Se calcula la distancia entre el labio superior (punto 51) y el labio inferior (punto 57). Si esta distancia supera un umbral, se considera que la boca está abierta.
   - Cuando la boca está abierta, el sistema inicia la secuencia de efectos de fuego.

  **4. Generación de Fuego.**  
   Cuando se detecta una boca abierta:
   - Se calcula una línea perpendicular desde el centro de la boca y se colocan emojis de fuego a lo largo de esta línea.
   - Los emojis de fuego se muestran de forma secuencial, aumentando hasta un límite de longitud, para simular una respiración de fuego.
   
   La orientación de la línea de fuego puede ajustarse usando las teclas `A` y `D` para rotar el ángulo a la izquierda o derecha, ajustando así la dirección de la “llama”.

  **5. Interfaz de Video en Tiempo Real.**  
   El sistema muestra en una ventana los efectos de máscara y fuego aplicados sobre el rostro en tiempo real. Además:
   - Se indican las opciones de interacción y se permite al usuario ajustar la dirección del fuego.
   - En caso de no detectar la boca o el rostro, se limpia la secuencia de emojis de fuego.

<div align="center">
    <!-- Ejemplo de Imagen de Efecto de Máscara y Fuego -->
    <div>
        <a href="filtro_dragon_fuego.JPG" target="_blank">
            <img src="./filtro_dragon_fuego.JPG" alt="Filtro de Dragón y Fuego" width="150">
        </a>
    </div>
</div>

## Ejecución y Controles
- **Para iniciar el filtro**, ejecute el archivo de código.
- **Controles**:
  - **A**: Rotar la línea de fuego a la izquierda.
  - **D**: Rotar la línea de fuego a la derecha.
  - **Q**: Salir del programa.
- La detección de la boca y la activación del fuego son automáticas; los efectos se apagan cuando la boca está cerrada.

## Procedimiento seguido para filtro festivo:

  **1. Carga de Recursos y Modelos.**  
  - Carga de recursos necesarios para la detección de emociones y la aplicación de efectos visuales.
  - Se carga la imagen 'corona.png', que representa una corona que se usará cuando se detecte la emoción de felicidad.
  - Además, se cargan todos los frames de animación desde la carpeta correspondiente para crear una animación que se superpondrá sobre toda la ventana.
    - Para poder visualizar el confeti, primero se tuvo que sacar cada fotograma del video.
    - Luego quitar el fondo de cada fotograma con la función previa al código de la práctica, para poder superponelo a la captura de la webcam.   
  - Se inicializan el detector de rostros y el predictor de puntos faciales utilizando el modelo preentrenado 'shape_predictor_68_face_landmarks.dat' de dlib.
  - Se establece la captura de video desde la cámara web.

  **2. Definición de Funciones Auxiliares.**
  - Función overlay_image_alpha permite superponer una imagen con transparencia sobre otra, manejando correctamente el canal alfa para preservar la transparencia.
  - La función is_smiling determina si la persona en el frame está sonriendo calculando la relación entre el ancho y el alto de la boca a partir de puntos faciales específicos.
  - La función crear_haz_cono_suave genera imágenes de haces de luz en forma de cono con bordes suaves para crear un efecto de iluminación.
  - La función overlay_conos selecciona aleatoriamente dos de estos haces y los superpone en el frame para simular un efecto de luces de discoteca.

  **3. Preparación de Efectos Visuales.**
  - Se captura un frame inicial para obtener las dimensiones de ancho y alto del video. Con esta información, se crean múltiples haces de luz posicionados y dirigidos de manera alternada a lo largo del ancho del frame, almacenándolos en una lista para su uso posterior. También se inicializa un contador de frames que se utilizará para controlar la secuencia de animación y los efectos visuales aplicados durante la ejecución.

  **4. Bucle Principal de Procesamiento de Video.**
  - El programa entra en un bucle que procesa cada frame capturado en tiempo real.
  - Convierte el frame a escala de grises para facilitar la detección de rostros utilizando el detector de dlib.
  - Por cada rostro detectado, se obtienen los 68 puntos faciales clave. Se utiliza la función is_smiling para determinar si la persona está sonriendo basándose en la geometría de su boca.
  - Si se detecta una sonrisa, se establece la emoción detectada como 'happy' y se aplican los efectos. Si no se detecta una sonrisa, la emoción se establece como 'neutral' y no se aplican efectos adicionales.

  **5. Visualización y Control de Salida.**
  - El frame procesado, con los efectos correspondientes aplicados, se muestra en una ventana titulada 'Filtro de Felicidad'.
  - Presionando la tecla 'q' se rompe el bucle y procede a finalizar la ejecución.

<div align="center">
    <!-- Ejemplo de GIF de Efecto de Máscara y Fuego desde Google Drive -->
    <div>
        <a href="https://drive.google.com/uc?export=view&id=1HzLMGbFDBw-HHOpDrYrPmM3OKETzzV95" target="_blank">
            <img src="https://drive.google.com/uc?export=view&id=1HzLMGbFDBw-HHOpDrYrPmM3OKETzzV95" alt="Efecto festivo en GIF" width="150">
        </a>
    </div>
</div>
