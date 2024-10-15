# Práctica 3 - Detección y reconocimiento de formas

  En esta práctica el objetivo es adquirir nociones para extraer información geométrica de objetos presentes en una imagen, con el fin de caracterizarlos y posteriormente ser capaz de identificarlos de forma automática en categorías.. Las áreas clave incluyen el análisis caractarístico de las formas que tienen los objetos como las monedas (Tarea 1) y los microplásticos (Tarea 2). El objetivo principal es darnos cuenta de que podemos utilizar relaciones geométricas, matemáticas o visuales para poder clasificarnos. En esta práctica también aprenderemos aspectos complicados en la detección de formas como lo son: el brillo, las sombras, el contraste e incluso en una tarea, el deterioro del objeto.
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
  **Objetivo:** Captura una o varias imágenes con monedas no solapadas. Tras visualizar la imagen, identifica de forma interactiva (por ejemplo haciendo clic en la imagen) una moneda de un valor determinado en la imagen (por ejemplo de 1€). Tras ello, la tarea se resuelve mostrando por pantalla el número de monedas y la cantidad de dinero presentes en la imagen. No hay restricciones sobre utilizar medidas geométricas o de color. ¿Qué problemas han observado?

  - Procedimiento seguido:

  1. Inicializamos las imágenes y las variables globales.

<!-- Filas de dos fotos cada una -->
<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="#práctica-3---detección-y-reconocimiento-de-formas" target="_blank">
            <img src="./VC - Prueba-Monedas1.jpg" alt="Imagen 1" width="300">
            <img src="./VC - Prueba-Monedas2.jpg" alt="Imagen 2" width="300">
        </a>
    </div>
</div>

<p>&nbsp;</p>

  2. Intentamos detectar el fondo de la imagen para posteriormente convertirlo a blanco y conseguir un mejor contraste entre el fondo y las monedas. Con esto se busca conseguir que la detección de las monedas sea más precisa.
  
  <!-- Filas de dos fotos cada una -->
<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="#práctica-3---detección-y-reconocimiento-de-formas" target="_blank">
            <img src="./VC - Pr3_Prueba-Monedas1-FondoBlanco.png" alt="Imagen 1" width="300">
            <img src="./VC - Pr3_Prueba-Monedas2-FondoBlanco.png" alt="Imagen 2" width="300">
        </a>
    </div>
</div>

<p>&nbsp;</p>

  3. A continuación comenzamos a detectar monedas. Para ello convertimos a una escala de griseses y y un filtro de desenfoque para reducir el ruido. Luego utilizamos la función cv2.HoughCircles() en la que configuramos los parámetros: resolución de acumulación (dp), la distancia mínima entre círculos (minDist) y el rango del radio. Si se detectan círculos los convierte a enteros y los guarda en una lista como tuplas (x, y, r), donde x y y son las coordenadas del centro y r es el radio del círculo.



  4. Una vez hemos detectado los círculos, procedemos a aplicarle dos filtros para buscar las monedas según su color. El primer filtro se encarga de encontrar las que son de color dorado (0.10€, 0.20€, 0.50€, 1€ y 2€). El otro filtro se encarga de buscar (0.01€, 0.02€ y 0.05€), es decir, las de color cobrizo. Se usa este enfoque para que los parámetros de color estén bien segmentados y evitar falsos positivos. Para encontrar el rango HSV, se cogieron 3 puntos relevantes correspondientes a cada categoría de monedas (Doradas o cobrizas), de forma que quedaran bien segmentadas en ambas imágenes. Finalmente se unen ambas detecciones en una misma estructura de datos.

### Detección de las monedas de color dorado:

  <!-- Filas de dos fotos cada una -->
<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="#práctica-3---detección-y-reconocimiento-de-formas" target="_blank">
            <img src="./VC - Pr3_Deteccion-Doradas_1.png" alt="Imagen 1" width="300">
            <img src="./VC - Pr3_Detección-Doradas_2.png" alt="Imagen 2" width="300">
        </a>
    </div>
</div>

<p>&nbsp;</p>

### Detección de las monedas de color cobrizo:

  <!-- Filas de dos fotos cada una -->
<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="#práctica-3---detección-y-reconocimiento-de-formas" target="_blank">
            <img src="./VC - Pr3_Deteccion-Cobrizas_1.png" alt="Imagen 1" width="300">
            <img src="./VC - Pr3_Deteccion-Cobrizas_2.png" alt="Imagen 2" width="300">
        </a>
    </div>
</div>

<p>&nbsp;</p>

  5. Para que la visualización sea más adecuada las detecciones se superponen a la imagen original dando la sensación de que las detecciones han sido sobre la imagen original sin haber aplicado transformaciones a la imagen.

 <!-- Filas de dos fotos cada una -->
<div align="center">
    <!-- Fila 1 -->
    <div>
        <a href="#práctica-3---detección-y-reconocimiento-de-formas" target="_blank">
            <img src="./VC - Pr3_Detección-Total1.png" alt="Imagen 1" width="300">
            <img src="./VC - Pr3_Detección-Total2.png" alt="Imagen 2" width="300">
        </a>
    </div>
</div>

<p>&nbsp;</p>

  6. Una vez tenemos capturadas todas las monedas posibles, clicaremos en la de 0.5€ que es la que se escogió de referencia y se clasificarán las monedas restantes en base a las siguientes fórmulas.

### Cálculo del Diámetro Detectado
La fórmula para calcular el diámetro detectado del círculo es:

$$ \text{diámetro detectado} = 2 \times r $$

### Cálculo de la Relación de Escala
La relación de escala entre el diámetro detectado y el diámetro real de la moneda de referencia es:

$$ \text{relacion escala} = \frac{\text{diámetro real 50c}}{\text{diámetro detectado}} $$

### Cálculo del Diámetro Real de las Monedas Detectadas
Con la relación de escala, se calcula el diámetro real de cualquier moneda detectada mediante:

$$ \text{diámetro real} = \text{diámetro detectado} \times \text{relacion escala} $$

<p>&nbsp;</p>


  8. Finalmente se muestran los resultados. En ellos se reflejan: el número de monedas detectadas, la suma total de las monedas detectadas y de cara a comprobar la fiabilidad del sistema el error producido entre la realidad de la suma del valor y el conteo de monedas real con respecto al obtenido.

### Resultados Imagen 1.

```
Reference coin radius (50 cents): 109
Scale ratio (mm/pixels): 0.11123853211009174
Total money in the image: 1.51 euros, Total real: 0.94
Resultados de la detección de monedas:
=====================================
Monedas de 1 céntimo detectadas: 2 --- reales: 3
Monedas de 2 céntimos detectadas: 2 --- reales: 3
Monedas de 5 céntimos detectadas: 1 --- reales: 1
Monedas de 10 céntimos detectadas: 2 --- reales: 1
Monedas de 20 céntimos detectadas: 1 --- reales: 1
Monedas de 50 céntimos detectadas: 2 --- reales: 1

Número de monedas detectadas: 10 --- Número de monedas reales:10
Relación de escala utilizada para la detección:
Relación de escala (mm/píxeles): 0.1112
```

### Resultados Imagen 2.

```
Reference coin radius (50 cents): 84
Scale ratio (mm/pixels): 0.14434523809523808
Total money in the image: 4.07 euros, Total real: 3.01
Resultados de la detección de monedas:
=====================================
Monedas de 1 céntimo detectadas: 3 --- reales: 3
Monedas de 2 céntimos detectadas: 2 --- reales: 4
Monedas de 5 céntimos detectadas: 4 --- reales: 2
Monedas de 10 céntimos detectadas: 1 --- reales: 1
Monedas de 20 céntimos detectadas: 1 --- reales: 1
Monedas de 50 céntimos detectadas: 1 --- reales: 1
Monedas de 1 euro detectadas: 1 --- reales: 2
Monedas de 2 euros detectadas: 1 --- reales: 0

Número de monedas detectadas: 14 --- Número de monedas reales:14
Relación de escala utilizada para la detección:
Relación de escala (mm/píxeles): 0.1443
```

  **Conclusiones:**
  Con esta tarea hemos comprobado que no es una tarea trivial el hecho de que el computador sea capaz de reconocer formas. En concreto se ve que en las dos imágenes de prueba, son factores determinantes: el estado de la moneda, el brillo que pueda tener la moneda por consecuencia de la luz, las sombras que se puedan generar también a causa de la luz, la presencia de una referencia para poder clasificar las monedas y la distancia a la que se encuentra la imagen. Estos son factores que pueden hacer que el reconocimiento de formas sea una tarea complicada.
