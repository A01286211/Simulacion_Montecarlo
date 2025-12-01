#STREAMLIT APPLICATION FOR PRUEBA.PY
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulación Montecarlo - Integral Cruda", layout="centered")

st.title("Simulación Montecarlo: Estimación de Integral (Método Crudo)")
st.write("Basado en `prueba.py`. Ingresa los parámetros y ejecuta la simulación.")

st.sidebar.header("Parámetros")
lim_inf_input = st.sidebar.text_input("Límite inferior (usar '-inf' para infinito)", value="-1")
lim_sup_input = st.sidebar.text_input("Límite superior (usar 'inf' para infinito)", value="1")
replicas = st.sidebar.number_input("Tamaño de muestra (réplicas)", min_value=1, max_value=200000, value=100, step=100)
N = st.sidebar.number_input("Número de corridas (N)", min_value=1, max_value=200000, value=10000, step=1000)

# Limitar infinitos
def parse_limits(lim_inf_str, lim_sup_str):
    if lim_inf_str.strip().lower() == "-inf":
        lower = -1e3
    else:
        lower = float(lim_inf_str)

    if lim_sup_str.strip().lower() == "inf":
        upper = 1e3
    else:
        upper = float(lim_sup_str)
    return lower, upper

def func_objetivo(x):
    return 1 / (np.exp(x) + np.exp(-x))

def Montecarlo_Examen(N, replicas, lower, upper):
    U = np.random.uniform(0, 1, (N, replicas))
    variables = lower + (upper - lower) * U
    evals = func_objetivo(variables)
    areas = (upper - lower) / replicas * np.sum(evals, axis=1)
    integral_crudo = np.mean(areas)
    return {
        "Valores_Aleatorios": variables,
        "Alturas_Funcion": evals,
        "Areas_Funcion": areas,
        "Integral_Crudo": integral_crudo,
    }

# Botón para ejecutar la simulación
run = st.sidebar.button("Ejecutar simulación")

if run:
    try:
        lower, upper = parse_limits(lim_inf_input, lim_sup_input)
        if lower >= upper:
            st.error("El límite inferior debe ser menor que el superior.")
        else:
            resultados = Montecarlo_Examen(N=N, replicas=replicas, lower=lower, upper=upper)

            st.subheader("Resultados")

            # Estimación de la integral
            st.write(f"Estimación de la integral (método crudo): `{resultados['Integral_Crudo']:.4f}`")

            # Mostrar valores aleatorio, alturas y áreas
            st.write("Para las primeras 5 réplicas:")
            valores_head = resultados["Valores_Aleatorios"][:5, :5]
            alturas_head = resultados["Alturas_Funcion"][:5, :5]
            areas_head = resultados["Areas_Funcion"][:5]

            st.write("Valores Aleatorios:")
            st.dataframe(valores_head)
            st.write("Alturas de la Función:")
            st.dataframe(alturas_head)
            st.write("Áreas bajo la Función:")
            st.dataframe(areas_head)

            # Estadísticas
            st.subheader("Estadísticas de Áreas")
            areas = resultados["Areas_Funcion"]
            st.write(
                {
                    "media": float(np.mean(areas)),
                    "desviacion_estandar": float(np.std(areas, ddof=1)) if len(areas) > 1 else 0.0,
                    "min": float(np.min(areas)),
                    "max": float(np.max(areas)),
                }
            )

            st.success("Simulación completada.")
    except ValueError:
        st.error("Por favor, ingresa límites válidos (números o '-inf'/'inf').")
else:
    st.info("Configura parámetros en la barra lateral y presiona 'Ejecutar simulación'.")