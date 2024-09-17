# Práctica 1 - Primeros pasos con OpenCV

El objetivo es comprender de manera práctica la representación de imágenes en escala de grises y en color, así como su modificación, visualización y tratamiento básico. Al finalizar la práctica, somos capaces de crear una imagen de un tamaño específico, acceder y modificar los valores de un píxel, dibujar formas gráficas simples sobre una imagen y acceder a los fotogramas de un video o de una captura de cámara.

## Autores
[![GitHub](https://img.shields.io/badge/GitHub-Javier%20Gómez%20Falcón-red?style=flat-square&logo=github)](https://github.com/GomFal)
[![GitHub](https://img.shields.io/badge/GitHub-Cristian%20Marrero%20Vega-blue?style=flat-square&logo=github)](https://github.com/XxMARRExX)

## Tecnologías
  - Python

## Librerías 
  - OpenCV
  - Matplotlib
  - NumPy

## TAREA 1: Crea una imagen, p.e. de 800x800 píxeles, con la textura del tablero de ajedrez

- **Paso 1. Crear imagen**: Se crea una imagen en blanco de tamaño 800x800 píxeles con tres canales (RGB).
- **Paso 2. Generar patrón de ajedrez**:
      Se rellenan cuadros de 100x100 píxeles en negro, alternando las posiciones para simular un tablero de ajedrez.
      Los bucles for se encargan de rellenar las posiciones alternas para los cuadros negros.
- **Paso 3. Dimensiones**: Se imprime el tamaño de la imagen resultante (800x800 con 3 canales).
- **Paso 4. Mostrar imagen**: Se visualiza la imagen generada usando matplotlib, mostrando el patrón de ajedrez.

<p>&nbsp;</p>

<!-- Filas de dos fotos cada una -->
<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="./P5/Readme Images/1.jpg" target="_blank">
            <img src="./P1/tablero.jpg" alt="Imagen 1" width="300">
        </a>
    </div>
</div>

## TAREA 2: Crear una imagen estilo Mondrian

- **Paso 1. Definir parámetros**: Establece tamaño de imagen, grosor de líneas y margen.
- **Paso 2. Crear imagen**: Crea una imagen blanca de 800x800 píxeles.
- **Paso 3. Dibujar líneas horizontales**: Añade tres líneas negras horizontales.
- **Paso 4. Dibujar líneas verticales**: Añade varias líneas negras verticales para formar una cuadrícula.
- **Paso 5. Dibujar rectángulos**: Rellena varias áreas de la cuadrícula con rectángulos de diferentes colores.
- **Paso 6 Visualizar imagen**: Muestra la imagen con matplotlib.
- **Paso 7 Guardar imagen**: Guarda la imagen como 'imagen.jpg'.

<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="./P5/Readme Images/1.jpg" target="_blank">
            <img src="./P1/imagen.jpg" alt="Imagen 1" width="300">
        </a>
    </div>
</div>

## TAREA 3: Resuelve una de las tareas previas (a elegir) con las funciones de dibujo de OpenCV 

- **Paso 1. Crear imagen: Se crea una imagen en blanco de tamaño 800x800 píxeles con tres canales (RGB).
- **Paso 2. Generar patrón de ajedrez**:
      Se rellenan cuadros de 100x100 píxeles en negro, alternando las posiciones para simular un tablero de ajedrez.
      Los bucles for se encargan de rellenar las posiciones alternas para los cuadros negros. **La diferencia
      es que esta vez se coloearán los cuadros con la función cv2.rectangle()**
- **Paso 3. Dimensiones**: Se imprime el tamaño de la imagen resultante (800x800 con 3 canales).
- **Paso 4. Mostrar imagen**: Se visualiza la imagen generada usando matplotlib, mostrando el patrón de ajedrez.

<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="./P5/Readme Images/1.jpg" target="_blank">
            <img src="./P1/tablero_opencv.jpg" alt="Imagen 1" width="300">
        </a>
    </div>
</div>

## TAREA 4: Modifica de forma libre los valores de un plano de la imagen

- **Paso 1. Captura de video**:  Se inicia la captura de video en tiempo real con la cámara.
- **Paso 2. Procesar cada cuadro:**:  Se lee cada cuadro de la cámara
- **Paso 3. Modificar un plano**: Se modifica un fragmento del canal azul (del píxel [0:100, 25:180]) asignándole el valor 1 (negro).
- **Paso 4. Crear un collage**: Se concatenan horizontalmente los canales rojo, verde y azul para formar una imagen combinada.
- **Paso 5. Mostrar imagen**: Se redimensiona el collage para ajustarse a la pantalla y se muestra en una ventana.
- **Paso 6 Finalización**:  El programa se detiene al presionar ESC, liberando los recursos de la cámara y cerrando las ventanas.


## TAREA 5.1: Pintar círculos en las posiciones del píxel más claro y oscuro de la imagen 

- **Paso 1. Captura de video**:  Se inicia la captura de video en tiempo real con la cámara.
- **Paso 2. Procesar cada cuadro:**:  Se lee cada cuadro y se convierte a escala de grises para simplificar el cálculo de los píxeles más claro y más oscuro.
- **Paso 3. Calcular píxeles más claro y más oscuro:**:  Se utilizan las funciones argmin() y argmax() para encontrar los índices de los píxeles más oscuro y más claro en la imagen.
- **Paso 4. Dibujar círculos**:  Se dibujan círculos en el cuadro original alrededor de los píxeles más oscuro (círculo blanco) y más claro (círculo negro).
- **Paso 5. Mostrar imagen**:  Se muestra el cuadro con los círculos en una ventana.
- **Paso 6. Mostrar collage**:  Se ensamblan las versiones coloreadas en un collage horizontal.
- **Paso 7 Finalización**:  El programa se detiene al presionar ESC, liberando los recursos de la cámara y cerrando las ventanas.


## TAREA 5.2: Pintar rectángulos sobre la zona 8x8 más clara y oscura

- **Paso 1. Captura de video**:  Se inicia la captura de video en tiempo real con la cámara.
- **Paso 2. Función de búsqueda**: Se define la función find_extreme_zone() para encontrar la zona 8x8 más clara o más oscura en la imagen en escala de grises.
- **Paso 3. Procesar cada cuadro**: Se lee cada cuadro y se convierte a escala de grises.
- **Paso 4. Buscar zonas más claras y oscuras**: Se utilizan las sumas de píxeles en ventanas de 8x8 para localizar las zonas más brillantes y más oscuras.
- **Paso 5. Dibujar rectángulos**: Se dibujan rectángulos verdes en la zona más clara y rojos en la zona más oscura del cuadro.
- **Paso 6. Mostrar imagen**: Se muestra el cuadro con los rectángulos en una ventana.
- **Paso 7 Finalización**:  El programa se detiene al presionar ESC, liberando los recursos de la cámara y cerrando las ventanas.


## TAREA 6: Llevar a cabo una propuesta propia de pop art

- **Paso 1. Captura de video**:  Se inicia la captura de video en tiempo real con la cámara.
- **Paso 2. Dimensiones de la cámara**:  Se obtienen las dimensiones originales del video y se reducen a la mitad.
- **Paso 3. Crear collage**:  Se crea una imagen vacía que contiene cinco veces el ancho del video para mostrar las variaciones.
- **Paso 4. Colores Pop Art**:  Se definen cinco colores (de amarillo a rojo) para aplicar un efecto de gradiente.
- **Paso 5. Procesar cada cuadro**:  Cada cuadro del video se redimensiona y se colorea con los tintes definidos, creando cinco versiones.
- **Paso 6. Mostrar collage**:  Se ensamblan las versiones coloreadas en un collage horizontal.
- **Paso 7. Finalización**:  El programa se detiene al presionar ESC, liberando los recursos.


