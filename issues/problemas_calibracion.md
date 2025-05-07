# Problemas con la Calibración Estéreo y Soluciones

## Problema Identificado

Hemos detectado problemas significativos en la calibración estéreo de las cámaras GoPro HERO11 Black:

1. **Error de calibración muy alto**: El error RMS (Root Mean Square) de la calibración estéreo es de 136.59, cuando idealmente debería ser menor a 1.0 píxel.

2. **Desalineación de cámaras**: Los datos de calibración muestran que las cámaras tienen:
   - Rotación significativa entre ellas (matriz R con valores altos)
   - Gran desplazamiento vertical (~520 mm en el eje Y)
   - Gran desplazamiento en profundidad (~518 mm en el eje Z)

3. **Fallo en generación de mapas de profundidad**: Como resultado, los mapas de disparidad y profundidad salen completamente negros/azules.

## Causas del Problema

La configuración actual de las cámaras no es adecuada para visión estéreo:

- **Posición incorrecta**: Para visión estéreo, las cámaras deberían estar montadas una al lado de la otra con ejes ópticos paralelos.
- **Parámetros de calibración inconsistentes**: Los valores de traslación indican que las cámaras están muy separadas en vertical y en profundidad.
- **Rectificación inadecuada**: La alta desalineación impide que el algoritmo de rectificación funcione correctamente.

## Solución: Requisitos para la Calibración

Para lograr una calibración estéreo exitosa:

### 1. Configuración Adecuada de Cámaras

- **Montaje en paralelo**: Ambas cámaras deben estar montadas una al lado de la otra, a la misma altura.
- **Distancia horizontal (baseline)**: Entre 10-20 cm entre los centros de las lentes.
- **Alineación horizontal**: Usar un nivel para asegurar que las cámaras estén perfectamente alineadas.
- **Soporte rígido**: Las cámaras deben estar firmemente fijadas para que no se muevan entre capturas.

### 2. Requisitos para Fotos de Calibración

- **Cantidad**: Mínimo 10-20 pares de imágenes del tablero de ajedrez.
- **Variedad de posiciones**: Capturar el tablero en diferentes posiciones dentro del campo visual.
- **Variedad de ángulos**: Incluir tomas con el tablero en diferentes ángulos.
- **Variedad de distancias**: Algunas fotos con el tablero cerca y otras más lejos.
- **Iluminación uniforme**: Evitar reflejos y sombras sobre el tablero.
- **Tablero rígido y plano**: Imprimir o pegar el patrón en una superficie completamente plana.
- **Visibilidad completa**: El tablero completo debe ser visible en ambas cámaras en cada par de imágenes.

### 3. Patrón de Calibración

- **Tamaño de tablero**: Preferiblemente 8x8 cuadros (7x7 esquinas internas).
- **Tamaño físico adecuado**: Cada cuadrado de 20-30 mm para buena detección.
- **Contraste alto**: Blanco puro y negro puro para mejor detección.
- **Superficie mate**: Evitar patrones brillantes que generen reflejos.

## Guía Detallada para la Sesión de Fotos de Calibración

### Preparación

1. **Preparación del tablero**:
   - Imprimir el patrón de ajedrez en alta calidad sin distorsión.
   - Pegarlo en una superficie rígida y perfectamente plana (cartón pluma, madera, acrílico).
   - Verificar que no tenga burbujas, arrugas o deformaciones.
   - Para crear un patrón personalizado: `python -m utils.pattern_generator --type chessboard --width 8 --height 8 --output tablero_calibracion.png`

2. **Preparación de las cámaras**:
   - Montar ambas cámaras en un soporte rígido a exactamente la misma altura.
   - Medir y anotar la distancia exacta entre los centros de las lentes (baseline).
   - Usar un nivel para asegurar que ambas cámaras estén perfectamente horizontales.
   - Configurar ambas cámaras con los mismos ajustes (resolución, FOV, exposición).
   - Desactivar cualquier estabilización de imagen o ajuste automático.

3. **Configuración del espacio**:
   - Elegir un área con buena iluminación uniforme (evitar luces directas o sombras fuertes).
   - Asegurar suficiente espacio para mover el tablero a diferentes distancias y ángulos.
   - Preparar un disparador sincronizado o método para capturar imágenes simultáneamente.

### Ejecución de la Sesión de Fotos

1. **Organización de archivos**:
   - Crear carpetas separadas para las imágenes de cada cámara (`data/calibration/left` y `data/calibration/right`).
   - Nombrar los archivos de forma consistente (ejemplo: `left01.jpg` y `right01.jpg` para el primer par).

2. **Serie de fotos sistemática**:
   - **Posición 1 - Centro cercano**: 
     - Sostener el tablero en el centro del campo visual a unos 50-80 cm de las cámaras.
     - Capturar 3-4 imágenes con pequeñas variaciones de ángulo (±15°).
   
   - **Posición 2 - Centro lejano**: 
     - Mover el tablero al centro a 1.5-2 metros de distancia.
     - Capturar 3-4 imágenes con variaciones de ángulo.
   
   - **Posición 3 - Esquina superior izquierda**: 
     - Sostener el tablero en la esquina superior izquierda del campo visual.
     - Capturar 2-3 imágenes a diferentes distancias.
   
   - **Posición 4 - Esquina superior derecha**: 
     - Repetir en la esquina superior derecha.
   
   - **Posición 5 - Esquina inferior izquierda**: 
     - Repetir en la esquina inferior izquierda.
   
   - **Posición 6 - Esquina inferior derecha**: 
     - Repetir en la esquina inferior derecha.
   
   - **Posición 7 - Ángulos pronunciados**: 
     - Capturar 2-3 imágenes con el tablero en ángulos más pronunciados (30-45°).

3. **Verificación durante la sesión**:
   - Revisar las imágenes después de cada grupo para confirmar que:
     - El tablero es completamente visible en ambas cámaras.
     - Las imágenes están enfocadas y bien iluminadas.
     - No hay reflejos o sombras que dificulten la detección de esquinas.
     - Las esquinas del tablero son claramente visibles.

4. **Captura adicional**:
   - Siempre tomar más fotos de las necesarias (25-30 pares).
   - Esto permite descartar imágenes problemáticas manteniendo suficientes para una buena calibración.

### Verificación Final y Procesamiento

1. **Revisión de las imágenes**:
   - Examinar cada par de imágenes para confirmar calidad y visibilidad del tablero.
   - Descartar imágenes borrosas, mal iluminadas o donde el tablero no sea completamente visible.
   - Asegurar que queden al menos 15-20 pares de buena calidad.

2. **Prueba de detección**:
   - Realizar una prueba de detección en algunas imágenes:
     ```bash
     python -m utils.advanced_detection --image data/calibration/left/left01.jpg --pattern chessboard --pattern_size 7x7 --visualize
     ```
   - Verificar que las esquinas se detecten correctamente.

3. **Ejecución de la calibración**:
   ```bash
   python -m calibration.gopro_calibration --left_imgs "data/calibration/left/*.jpg" --right_imgs "data/calibration/right/*.jpg" --output_dir results --baseline 184 --debug
   ```

4. **Evaluación de resultados**:
   - Error RMS objetivo: < 1.0 píxel
   - Matriz R: valores cercanos a la identidad
   - Vector T: valores pequeños en Y y Z (< 10mm idealmente)

## Alternativas si la Recalibración No es Posible

1. **Usar Estimación de Profundidad Monocular**: 
   - Funciona con una sola cámara usando el modelo MiDaS.
   - Menos preciso pero más flexible.
   - No requiere configuración estéreo.

2. **Usar Marcadores ArUco en lugar del Tablero de Ajedrez**:
   - Los marcadores ArUco son más robustos a condiciones de iluminación difíciles.
   - Funcionan mejor con lentes gran angular como las de GoPro.
   - Comando: `python -m calibration.aruco_calibration --generate_board --output_dir data/calibration`

## Conclusión

Los problemas actuales se deben principalmente a la configuración física de las cámaras. Para lograr una estimación de profundidad estéreo precisa (>95% de exactitud), es fundamental remontarlas correctamente y repetir el proceso de calibración siguiendo las pautas mencionadas.