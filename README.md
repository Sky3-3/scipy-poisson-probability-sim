# Proyecto Python: Modelado Probabilístico y Simulación Estocástica con la Distribución de Poisson

Este repositorio contiene un proyecto práctico desarrollado en Python utilizando las librerías **SciPy** (módulo `stats`) y **NumPy** enfocado en el modelado estadístico de variables aleatorias discretas mediante la Distribución de Poisson. El script simula el monitoreo de fallas en una línea de producción industrial basándose en una tasa promedio de ocurrencia ($\lambda = 7$ defectos por día), calcula probabilidades puntuales y acumuladas, y evalúa el comportamiento de un conjunto de datos sintético generado a lo largo de un ciclo anual ($365$ días) para validar de forma empírica los teoremas asintóticos de la media y percentiles de control de calidad.

---

## Código Python del Proyecto

El programa calcula las probabilidades de ocurrencia de eventos discretos independientes y realiza un análisis de agregación sobre el dataset simulado:

```python
import scipy.stats as stats
import numpy as np

# --- Task Group 1: Análisis de Probabilidad Teórica ---
# Configuración del parámetro de tasa (lambda): Promedio de defectos esperados por día
lam = 7

# Task 2: Función de Masa de Probabilidad (PMF)
# Calcula la probabilidad exacta de observar una cantidad de eventos igual a la media teórica
print("Probabilidad exacta de observar 7 defectos: " + str(stats.poisson.pmf(lam, lam)))

# Task 3: Función de Distribución Acumulada (CDF)
# Calcula la probabilidad de observar 4 o menos defectos en un día determinado
print("Probabilidad de 4 o menos defectos: " + str(stats.poisson.cdf(4, lam)))

# Task 4: Probabilidad del Complemento
# Calcula la probabilidad de observar estrictamente más de 9 defectos en un día
print("Probabilidad de más de 9 defectos: " + str(1 - stats.poisson.cdf(9, lam)))

# --- Task Group 2: Simulación Concreta de un Ciclo Anual (365 días) ---
# Task 5: Generación de Variables Aleatorias (RVS)
# Simula el registro diario de fallas de la fábrica durante un año completo
year_defects = stats.poisson.rvs(lam, size=365)

# Task 6: Muestreo de control de la simulación (primeros 20 días)
print("\nRegistro de defectos (Primeros 20 días):")
print(year_defects[0:20])

# Tasks 7 & 8: Comparativa de Totales (Teórico vs Real)
print("\nTotal de defectos esperado teóricamente: " + str(lam * 365))
print("Total de defectos registrados en la simulación: " + str(sum(year_defects)))

# Task 9: Validación empírica del valor esperado
print("Media aritmética de la muestra simulada: " + str(np.mean(year_defects)))

# Tasks 10 & 11: Análisis del Valor Máximo Registrado
max_defects = year_defects.max()
print(f"Máximo número de defectos registrados en un solo día: {max_defects}")
print("Probabilidad teórica de superar dicho valor máximo: " + str(1 - stats.poisson.cdf(max_defects, lam)))

# --- Extra Bonus: Análisis de Percentiles de Control ---
# Task 12: Probabilidad acumulada en un umbral flotante
print("\nEvaluación de límites de control:")
print(stats.poisson.cdf(0.9, lam))

# Task 13: Proporción empírica de días que superan el percentil 90%
# Utiliza la función ppf (Percent Point Function) para hallar el valor crítico de corte
val_critico_90 = stats.poisson.ppf(0.9, lam)
proporcion_dias_criticos = sum(year_defects > val_critico_90) / len(year_defects)
print(f"Proporción de días en el año con fallas sobre el percentil 90% (Corte: {val_critico_90}): " + str(proporcion_dias_criticos))

```

---

## Fundamentos Matemáticos y Comportamiento de la Distribución

La distribución de Poisson es un modelo de probabilidad discreta que expresa la probabilidad de que ocurra un número determinado de eventos independientes en un intervalo fijo de tiempo o espacio, conociendo únicamente su tasa media de aparición ($\lambda$).

### 1. Ecuación General de la Función de Masa (PMF)

La probabilidad teórica de que ocurran exactamente $k$ eventos responde a la fórmula exponencial:

$$P(X = k) = \frac{\lambda^k \cdot e^{-\lambda}}{k!}$$

Donde $e$ es la base del logaritmo natural y $k!$ representa el factorial del número de eventos. El script utiliza esta relación para comprobar el comportamiento en el punto máximo de la curva ($\lambda=7$).

### 2. Resultados de la Simulación y Ley de los Grandes Números

A lo largo de las 365 iteraciones del vector estocástico, los datos sufren fluctuaciones debidas al azar (como registrar picos atípicos aislados de fallas). Sin embargo, al aplicar agregaciones estadísticas se valida la consistencia del modelo:

* **Convergencia de la Media:** El promedio aritmético obtenido sobre la muestra simulada (`np.mean(year_defects)`) tiende a aproximarse estrechamente al parámetro teórico original ($\lambda = 7.0$), cumpliendo con la **Ley de los Grandes Números**.
* **Proporción Crítica del Percentil:** La función `.ppf(0.9)` actúa como el inverso de la CDF, determinando cuál es el número de defectos máximo para quedar cubierto bajo el $90\%$ de los días estándar. Al evaluar la cantidad real de días que rompieron ese umbral protector, la proporción muestral resultante converge de forma estable cerca del $10\%$ restante esperado.

---

## Conceptos Técnicos Aplicados

* **Función de Masa de Probabilidad (PMF)**: Función integrada que calcula la probabilidad exacta de que una variable aleatoria discreta tome un valor numérico específico de la escala.
* **Función de Distribución Acumulada (CDF)**: Operador que calcula la sumatoria de las probabilidades acumuladas desde el límite inferior hasta el punto de control provisto ($P(X \le k)$), fundamental para evaluar rangos de tolerancia.
* **Percent Point Function (`ppf`)**: Conocida como la función cuantil o inversa de la CDF. Toma una probabilidad objetivo (ej: `0.90`) y devuelve el valor entero de la variable aleatoria que acumula dicha área bajo la curva.
* **Generación de Muestras Aleatorias (`rvs`)**: Algoritmo generador pseudoaleatorio basado en un modelo matemático subyacente que genera vectores de datos sintéticos independientes para realizar simulaciones de Montecarlo o auditorías de procesos de control de calidad.

