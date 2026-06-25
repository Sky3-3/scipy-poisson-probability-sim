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
