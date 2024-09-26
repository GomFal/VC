# Práctica 2 - Funciones básicas de OpenCV

El objetivo de esta actividad es aplicar y comprender técnicas fundamentales de procesamiento de imágenes. Las áreas clave incluyen el análisis de píxeles, la detección de bordes, y la visualización de propiedades de las imágenes. El objetivo principal es mejorar la comprensión de estos conceptos básicos mediante la implementación de algoritmos que permiten detectar bordes, contar valores de intensidad de píxeles y visualizar características relevantes de una imagen.
## Autores
[![GitHub](https://img.shields.io/badge/GitHub-Javier%20Gómez%20Falcón-red?style=flat-square&logo=github)](https://github.com/GomFal)
[![GitHub](https://img.shields.io/badge/GitHub-Cristian%20Marrero%20Vega-blue?style=flat-square&logo=github)](https://github.com/XxMARRExX)

## Tecnologías
  - Python

## Librerías 
  - OpenCV
  - Matplotlib
  - NumPy

## TAREA 1: Detección de Bordes con Canny
  **Objetivo:** Aplicar el algoritmo de detección de bordes Canny en una imagen para identificar las áreas de cambio de intensidad.
  - Convertir la imagen a escala de grises.
  - Usar la función cv2.Canny() para detectar bordes en la imagen.
  - Visualizar los bordes detectados y analizar la densidad de los bordes en la imagen.
  - Contar los píxeles correspondientes a los bordes y extraer información relevante de la imagen.
**Resultado:**
    - La fila con mayor porcentaje de píxeles blancos tuvo un 42%. Se encuentra en las primeras filas de la imagen.
    - Existen solo 2 filas por encima del 95% de píxeles blancos por fila.
      

<p>&nbsp;</p>

<!-- Filas de dos fotos cada una -->
<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="./tablero.jpg" target="_blank">
            <img src="./tablero.jpg" alt="Imagen 1" width="300">
        </a>
    </div>
</div>

## TAREA 2: Detección de Bordes con Sobel
  **Objetivo:** Aplicar el algoritmo de detección de bordes Sobel en una imagen para identificar las áreas de cambio de intensidad.
    - Reducir ruido de la imagen original con GaussianBlur()
    - Calcula los bordes en las direcciones horizontal y vertical y combina los resultados con Sobel.
    - Umbraliza la imagen de Sobel, marcando los bordes como píxeles blancos.
    - Cuenta los píxeles blancos en cada fila y columna, y normaliza los resultados.
    - Calcula cuántas filas y columnas tienen más del 95% de píxeles blancos.
    
  **Resultado: **
    - Se obtiene un valor máximo de 0.31% de píxeles blancos en la imagen **Sobel**
    - Hay 3 filas con píxeles blancos que se encuentran por encima del 95%, siendo el 100% la fila con mayor cantidad de píxeles blancos
    - Existe 1 columna por encima del 95% 
    Comparado con Canny, debido al umbralizado hay mayor claridad en los bordes. Existe un menor porcentaje de pixeles     blancos en la imagen de Sobel.


<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="./imagen.jpg" target="_blank">
            <img src="./imagen.jpg" alt="Imagen 1" width="300">
        </a>
    </div>
</div>

## TAREA 3: Show de Conocimientos.  
  **Objetivos:** Capturar la webcam en tiempo real y mostrar un collage en el que se aplican varios efectos de procesamiento de imágenes: tinte sepia, escala de grises, suavizado, detección de bordes y umbralización.
  - Crear clasificador en cascada para detección de caras.
  - Recorrer la imagen, en este caso el frame, usando el detector
  - Aplicar un cuadrado y un texto sobre la cara detectada


<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="./tablero_opencv.jpg" target="_blank">
            <img src="./tablero_opencv.jpg" alt="Imagen 1" width="300">
        </a>
    </div>
</div>

## TAREA 4: Inspirarse en 3 vídeos de tratamiento de imágenes [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy), [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared) y [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared) para crear una reinterpretación de alguna idea obtenida de esos vídeos.
  **Objetivos:** A través de la Webcam detectar caras usando un modelo preentrenado y superponer una imagen en las          caras detectadas.

- **Paso 1. Captura de video**:  Se inicia la captura de video en tiempo real con la cámara.
- **Paso 2. Procesar cada cuadro**:  Se lee cada cuadro de la cámara
- **Paso 3. Modificar un plano**: Se modifica un fragmento del canal azul (del píxel [0:100, 25:180]) asignándole el valor 1 (negro).
- **Paso 4. Crear un collage**: Se concatenan horizontalmente los canales rojo, verde y azul para formar una imagen combinada.
- **Paso 5. Mostrar imagen**: Se redimensiona el collage para ajustarse a la pantalla y se muestra en una ventana.
- **Paso 6 Finalización**:  El programa se detiene al presionar ESC, liberando los recursos de la cámara y cerrando las ventanas.

<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="./resultado_Tarea_4.JPG" target="_blank">
            <img src="./resultado_Tarea_4.JPG" alt="Imagen 4" width="300">
        </a>
    </div>
</div>

