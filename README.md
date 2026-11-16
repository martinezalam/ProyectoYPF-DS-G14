# Proyecto Fundación YPF - Curso Data Science + IA.

# Venteo y quema de gas: ubicación, estación y pérdida significativa.

## Grupo 14

**Miembros:**

* Martínez, Alam / Ingeniero de Petróleos / martinezalam19@gmail.com

* Quinteros, Julián Ezequiel / Ingeniero de Petróleos / julian15quinteros@gmail.com

## Resumen Ejecutivo

*A continuación se presenta el resumen de los puntos clave del proyecto:*

**Resumen del proyecto**
Este proyecto analiza datos de producción de gas natural para identificar patrones asociados a una pérdida significativa del recurso (cuando el venteo y la quema superan el 10% del volumen total). Al cambiar la unidad de análisis de un simple renglón administrativo a una celda agregada por "ubicación × mes", el estudio busca caracterizar el comportamiento de la disposición del gas de manera temporal y geográfica. Para lograrlo, se implementan técnicas de Machine Learning tanto supervisadas como no supervisadas que permiten descubrir patrones subyacentes en el manejo de este hidrocarburo, buscando asociaciones sin afirmar causalidad.

**Origen del dataset**
El conjunto de datos original (`OGORBcsv.csv`) fue obtenido de la Office of Natural Resources Revenue (ONRR), perteneciente al Departamento del Interior de los Estados Unidos (U.S. Department of the Interior). El mismo contiene registros mensuales detallados sobre la producción y el destino de los hidrocarburos extraídos (gas y petróleo) en diversas clases de tierras (federales, indígenas, offshore) abarcando el período desde enero de 2015 hasta mayo de 2025.

**Objetivos**
El objetivo principal de la investigación es determinar qué ubicaciones geográficas y épocas del año están más fuertemente asociadas con un mes de producción que presente una tasa de pérdida significativa de gas. Secundariamente, el proyecto se propone desarrollar un flujo de trabajo reproducible que permita caracterizar los destinos del gas, evaluar qué variables aportan información relevante, y comparar el rendimiento de distintos modelos predictivos utilizando una validación estrictamente temporal para evitar la fuga de información.

**Descubrimientos en el Análisis Exploratorio**
Durante la fase exploratoria, descubrimos que era crucial filtrar y trabajar únicamente con volúmenes de gas (Mcf) para evitar sumar petróleo y gas, y que los 33 códigos de disposición debían reagruparse en categorías útiles (venta, venteo, quema). Además, notamos que los datos geográficos faltantes no eran aleatorios, sino que dependían del tipo de tierra (ej. registros offshore o tierras indígenas). Finalmente, detectamos una gran cantidad de valores atípicos en el volumen que no correspondían a errores, sino a pozos de altísima producción, lo que justificó analizar la pérdida como una tasa porcentual.

**Detalle de los modelos ajustados**
Para la predicción de pérdidas significativas, implementamos una partición temporal (entrenamiento 2015-2023; verificación 2024-2025) y evaluamos un modelo Dummy, Random Forest, Regresión Lineal sobre la tasa, y Regresión Logística. La Regresión Logística (con hiperparámetro C ajustado) resultó ser el modelo más robusto, alcanzando un ROC-AUC de 0.981 y detectando casi todas las celdas de alta pérdida sin falsas alarmas al 50%. Complementariamente, el análisis no supervisado (PCA y K-Means con k=6) reveló que las celdas no se agrupan estrictamente por su ubicación, sino que forman clusters basados en perfiles combinados de volumen manejado y proporción de pérdida.

## Documentación Extendida del Proyecto

### Pregunta de investigación

¿Qué **ubicación** y **época del año** se asocian con que un mes de producción tenga una **pérdida significativa** de gas (venteo + quema superior al 10 % del volumen total)?

El análisis busca *asociaciones* en datos observacionales. No se afirma causalidad.

### Alcances del proyecto

* Se analiza solo **gas (Mcf)**. Petróleo y gas no se suman, porque tienen unidades distintas.

* La unidad de análisis es la **celda ubicación × mes**, no en un renglón administrativo.

* La variable objetivo es `alta_perdida`: vale 1 si la tasa de pérdida de la celda supera el 10 %.

* El año **no se usa como predictor**. Se usa solo para separar entrenamiento y verificación.

### Objetivos Detallados

Desarrollar un proyecto de Machine Learning reproducible que pueda:

1. Caracterizar la disposición del gas producido (venta, venteo y quema) según ubicación y época del año.

2. Evaluar qué variables aportan información para distinguir la pérdida significativa.

3. Comparar modelos de referencia (línea base, modelos interpretables y no lineales) con validación temporal.

4. Interpretar los resultados y explicitar las limitaciones.

### Dataset

* *Archivo:* ⁠ OGORBcsv.csv ⁠

* *Fuente:* Office of Natural Resources Revenue (ONRR), U.S. Department of the Interior. Recuperado de: https://revenuedata.onrr.gov/downloads/production-by-disposition/

* *Granularidad:* mensual, por fecha de producción, tierra, ubicación, hidrocarburo y destino del volumen.

* *Cobertura temporal:* enero de 2015 a mayo de 2025. **2025 es un año parcial** (enero a mayo).

* *Dimensiones originales:* 470.830 filas × 11 columnas.

#### Columnas originales

| Columna | Descripción | 
| ----- | ----- | 
| ⁠ Production Date ⁠ | Mes de producción (formato MM/DD/YYYY, día siempre 01). | 
| ⁠ Land Class ⁠ | Clase de tierra (por ejemplo, federal, tribal). | 
| ⁠ Land Category ⁠ | Categoría de tierra (onshore u offshore). | 
| ⁠ State ⁠ | Estado de EE. UU. Vacío para registros offshore. | 
| ⁠ County ⁠ | Condado. | 
| ⁠ FIPS Code ⁠ | Código FIPS del condado. | 
| ⁠ Offshore Region ⁠ | Región offshore (Alaska, Golfo, Pacífico). | 
| ⁠ Commodity ⁠ | Tipo de hidrocarburo y unidad (⁠ Gas (Mcf) ⁠ u ⁠ Oil (bbl) ⁠). | 
| ⁠ Disposition Code ⁠ | Código numérico del destino del volumen. | 
| ⁠ Disposition Description ⁠ | Descripción textual del destino. | 
| ⁠ Volume ⁠ | Volumen reportado. Viene como texto con separadores de miles y puede tener valores negativos. | 

### Preparación de datos (resumen detallado)

* *Volumen:* se convierte a numérico quitando las comas. Los valores negativos son ajustes contables (códigos de disposición 13, 25 y 42). El archivo completo tiene 13.036 negativos, pero ninguno queda en el subconjunto de gas que se analiza.

* *Fechas:* se convierten a `datetime` con formato explícito. A partir de ellas se derivan año, mes y estación.

* *Destinos del volumen:*

  * *Venta:* códigos 1, 4, 5, 6, 7, 9, 15, 43.

  * *Venteo:* códigos 61, 62, 63, 68, 69.

  * *Quema:* códigos 21, 22, 33, 66, 67.

  * *Otros destinos* (transferencias, inyección, ajustes, etc.): se excluyen, porque no son venta ni pérdida.

* *Faltantes:* se etiquetan según su causa, verificada con tablas cruzadas:

  * `Offshore Region` vacío en registros onshore → `No aplica (onshore)`.

  * `State` y `County` vacíos en registros offshore → `No aplica (offshore)`.

  * `State` y `County` vacíos en tierra indígena → `No reportado (tierra indígena)`.

  * No se imputan con la moda ni con la media, para no inventar información.

* *Ubicación:* se construye a partir de la región offshore o del estado. Los estados con menos de 1.500 registros se agrupan en `Otros estados`, para que las estimaciones por estado sean estables.

* *Resultado:* 68.964 registros de gas, sin duplicados exactos. Se agregan en 2.000 celdas (16 ubicaciones × 125 meses).

### Método

1. *Análisis exploratorio:* formato numérico, unidades, fechas, códigos de disposición, faltantes, valores sin información y outliers (método del IQR, que se conservan porque son pozos productivos reales).

2. *Target:* tasa de pérdida = volumen perdido / volumen total de la celda. `alta_perdida = 1` si la tasa supera el 10 %.

3. *Partición temporal:* entrenamiento 2015–2023 (1.728 celdas) y verificación 2024–mayo 2025 (272 celdas). La validación cruzada también respeta el orden temporal (`TimeSeriesSplit`).

4. *Predictores:* ubicación y estación del año, codificados con variables dummy (one-hot) con categorías de referencia (WY e Invierno).

5. *Modelos supervisados:*

   * Línea base (DUMMY).

   * Regresión logística, con el hiperparámetro C ajustado por validación cruzada temporal.

   * RANDOM FOREST.

   * RATE BY CELL (techo con estas variables).

   * Regresión sobre la tasa, clasificada con el mismo umbral del 10 %.

6. *Evaluación:* los modelos se comparan en validación cruzada temporal con entrenamiento. La verificación 2024–2025 se usa una sola vez al final. Se estima incertidumbre con bootstrap por bloques de meses, y se hace un análisis de sensibilidad al umbral (5 %, 10 % y 15 %).

7. *Análisis no supervisado (descriptivo):* PCA sobre el perfil de cada celda, K-Means con la elección de k por codo y Silhouette, y DBSCAN como exploración de meses atípicos. Las variables del perfil no incluyen la etiqueta.

### Resultados principales

#### Clasificación de pérdida significativa (verificación 2024–mayo 2025: 272 celdas, 53 positivas)

| Modelo | ROC-AUC | PR-AUC | Recall | Precisión | 
| ----- | ----- | ----- | ----- | ----- | 
| DUMMY (BASELINE) | 0,500 | 0,195 | 0,000 | – | 
| LOGISTIC REGRESSION | 0,981 | 0,975 | 0,962 | 1,000 | 
| RANDOM FOREST | 0,981 | 0,976 | 0,962 | 1,000 | 
| RATE BY CELL (CEILING) | 0,986 | 0,977 | 0,962 | 1,000 | 
| REGRESSION ON RATE (umbral 0,10) | 0,989 | 0,980 | 0,962 | 0,750 | 

La regresión logística es el modelo principal: rinde igual que los modelos más complejos y no genera falsas alarmas. No detecta 2 de las 53 celdas de alta pérdida.

#### Sensibilidad al umbral

| Umbral | Positivos en verificación | ROC-AUC | PR-AUC | 
| ----- | ----- | ----- | ----- | 
| 5 % | 84 | 0,883 | 0,827 | 
| 10 % | 53 | 0,981 | 0,975 | 
| 15 % | 50 | 1,000 | 0,998 | 

El resultado depende del umbral. Por eso el umbral del 10 % lo representamos junto a esta tabla.

#### Análisis no supervisado

* *PCA:* dos componentes explican el 95,7 % de la variación. El problema tiene, en la práctica, dos dimensiones: volumen total frente a tasa, y volumen perdido.

* *K-Means:* el Silhouette elige k = 6 (0,577 en entrenamiento y 0,527 en verificación). El codo no es nítido, así que la elección es orientativa.

* *Los clusters no son las ubicaciones:* el Adjusted Rand Index contra la ubicación es 0,28. Los clusters agrupan por perfil de volumen y pérdida.

* *Estabilidad:* el 96,6 % de los meses de cada ubicación cae en su cluster dominante en 2015–2023, y el 86,8 % en 2024–2025. Las excepciones son CO y LA, que cambian de grupo por cambios de **volumen**, con la tasa de pérdida casi igual.

* *DBSCAN* (exploratorio): 18 celdas de ruido (1 %), concentradas en Offshore Pacific. Son meses atípicos.
