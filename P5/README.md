## Proyecto de Filtro Facial: Máscara de Dragón y Fuego

Este proyecto implementa un filtro facial que coloca una máscara de dragón sobre la cara detectada en tiempo real. Además, cuando el sistema detecta que la boca está abierta, se generan emojis de fuego que simulan una respiración de fuego, siguiendo la posición de la boca y un ángulo ajustable. La detección facial y del estado de la boca se logra mediante el uso de la librería `dlib` para reconocimiento facial, y se visualizan en un flujo de video en tiempo real.

## Autores
[![GitHub](https://img.shields.io/badge/GitHub-Javier%20Gómez%20Falcón-red?style=flat-square&logo=github)](https://github.com/GomFal)
[![GitHub](https://img.shields.io/badge/GitHub-Cristian%20Marrero%20Vega-blue?style=flat-square&logo=github)](https://github.com/XxMARRExX)

## Tecnologías
  - Python

## Librerías 
  - OpenCV
  - dlib
  - NumPy

## Dataset de Detección Facial
Este proyecto utiliza el modelo preentrenado `shape_predictor_68_face_landmarks.dat` para la detección de 68 puntos faciales clave. Este modelo se emplea para detectar la posición de la boca y otros puntos de referencia para alinear la máscara de dragón y posicionar los efectos de fuego.

## Procedimiento seguido:
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

Este proyecto ilustra el uso de técnicas de superposición de imágenes y tracking de puntos faciales para crear un filtro visual interactivo que responde en tiempo real.

