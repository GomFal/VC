## Práctica 4. Reconocimiento de matrículas

En esta práctica, se implementa un sistema de detección y reconocimiento de objetos en video enfocado en la detección de personas, vehículos (coches y guaguas) y reconocimiento de matrículas. La práctica combina técnicas de detección de objetos mediante YOLOv5, reconocimiento óptico de caracteres (OCR) usando EasyOCR, y tracking de objetos para contar y registrar objetos únicos en cada fotograma. Esto ofrece una experiencia práctica en el uso de modelos de detección, OCR, y técnicas de preprocesamiento para mejorar la precisión en escenarios de video.

## Autores
[![GitHub](https://img.shields.io/badge/GitHub-Javier%20Gómez%20Falcón-red?style=flat-square&logo=github)](https://github.com/GomFal)
[![GitHub](https://img.shields.io/badge/GitHub-Cristian%20Marrero%20Vega-blue?style=flat-square&logo=github)](https://github.com/XxMARRExX)

## Tecnologías
  - Python

## Librerías 
  - OpenCV
  - NumPy
  - Torch
  - EasyOCR
  - Ultralytics

## Dataset para el Modelo de detección de matrículas.

Para el modelo específico de detección de matrículas, se utilizó un [Dataset](https://alumnosulpgc-my.sharepoint.com/:f:/g/personal/cristian_marrero104_alu_ulpgc_es/EuPgtSUUUHFLn-G3qW_PqhkBr2piUigs-SEoNBzKI3VzHA?e=9Uy6d2)
 personalizado que fue generado manualmente. Las imágenes del dataset fueron recopiladas de fuentes como Facebook Marketplace y varias páginas de ventas de vehículos.

Distribución de Datos: El dataset fue dividido en:
- 70% para entrenamiento
- 20% para pruebas (test)
- 10% para validación.
  
Esta distribución asegura un buen balance entre el entrenamiento del modelo y la capacidad de generalización para nuevas matrículas.
Este modelo personalizado permite que el sistema sea más preciso al detectar matrículas en condiciones de video real, lo cual es crucial para el procesamiento de OCR posterior.


## Procedimiento seguido:
  **1. Inicialización del Modelo y Variables Globales.** YOLO11n es el modelo usado para detectar las personas y coches, y el modelo con el que se entrenó para detectar las matrículas.
  
  **2. En cada fotograma del vídeo**, se utiliza el modelo YOLO para detectar personas, coches y autobuses. Las detecciones incluyen la clase de objeto, las coordenadas de la caja delimitadora, y el nivel de confianza.
  
  **3. Se asigna un identificador único (ID)** a cada objeto para evitar duplicación de conteo, asegurando que cada objeto detectado se cuente una sola vez mediante técnicas de tracking.
  
  **4. Cuando se detecta un vehículo** (coche o autobús), se extrae la región de interés (ROI) correspondiente a la ubicación esperada de la matrícula. Esta región se procesa (convertida a escala de grises, aumentada en contraste, y umbralizada) para mejorar la precisión del OCR.
  
  **5. Se utiliza EasyOCR** para el reconocimiento de texto en la imagen de la matrícula. El modelo de OCR procesa esta región de interés y, en caso de éxito, extrae el texto de la matrícula, que se almacena junto con las demás detecciones.
  
  **6. Cada detección** (personas, vehículos y matrículas) se registra en el archivo CSV resultados_detalles.csv. Los datos registrados incluyen:
       - Tipo de objeto detectado.
       - Nivel de confianza de la detección.
       - ID de tracking único.
       - Coordenadas de la caja delimitadora.
       - Texto de la matrícula, si se detecta.
  
  **8. Se genera un vídeo de salida** en el cual se dibujan las cajas delimitadoras y se muestran las anotaciones relevantes, como el tipo de objeto y la matrícula reconocida (cuando disponible).


