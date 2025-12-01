import numpy as np

lim_inf = input("Ingrese el límite inferior del intervalo de integración (o '-inf' para infinito): ")
lim_sup = input("Ingrese el límite superior del intervalo de integración (o 'inf' para infinito): ")
n = int(input("Ingrese el tamaño de muestra (réplicas): "))

# encerrar infinitos en rango [-10e6, 10e6]
if lim_inf.lower() == '-inf':
    lim_inf = -1e6
else:
    lim_inf = float(lim_inf)

if lim_sup.lower() == 'inf':
    lim_sup = 1e6
else:
    lim_sup = float(lim_sup)

def func_objetivo(x):
    return 1/(np.exp(x) + np.exp(-x))


def Montecarlo_Examen(N=10000, replicas=n, lower=lim_inf, upper=lim_sup):
    
    # Se crea muestra (replicas), de tamaño n, uniforme en [0,1]
    U = np.random.uniform(0, 1, (N, replicas))

    # Transformación lineal a [lower, upper]
    variables = lower + (upper - lower) * U

    # Evaluar función objetivo en las variables generadas
    evals = func_objetivo(variables)

    # Se calculan las areas
    areas = (upper - lower) / replicas * np.sum(evals, axis=1)

    # Estimacion de la integral por el método Crudo
    integral_crudo = np.mean(areas)
    
    return {
        "Valores_Aleatorios": variables,
        "Alturas_Funcion": evals,
        "Areas_Funcion": areas,
        "Integral_Crudo": integral_crudo,
    }

resultados = Montecarlo_Examen(N=10000)
print(f"Valores Aleatorios (primeras 5 réplicas):\n{resultados['Valores_Aleatorios'][:5]}")
print(f"Alturas de la Función (primeras 5 réplicas):\n{resultados['Alturas_Funcion'][:5]}")
print(f"Áreas bajo la Función (primeras 5 réplicas):\n{resultados['Areas_Funcion'][:5]}")
print(f"Estimación de la Integral por el Método Crudo: {resultados['Integral_Crudo']}")