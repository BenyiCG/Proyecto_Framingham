
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 1. Configuración de la página ---
st.set_page_config(page_title="Predicción de Riesgo Coronario", layout="wide")

# --- 2. Cargar el modelo entrenado ---
@st.cache_resource
def cargar_modelo():
    ruta = 'model/pipeline_framingham.pkl'
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None

modelo = cargar_modelo()

# --- 3. Función para cargar datos (solo para EDA) ---
@st.cache_data
def cargar_datos_eda():
    # Nota: Deberás subir 'framingham.csv' a tu repositorio de GitHub
    ruta_csv = 'framingham.csv' 
    if os.path.exists(ruta_csv):
        return pd.read_csv(ruta_csv)
    return None

df = cargar_datos_eda()

# --- 4. Interfaz de Usuario ---
st.title("🩺 Predicción de Riesgo de Enfermedad Coronaria a 10 años")
st.markdown("""
**Universidad de la Costa** | **Minería de Datos**
Esta aplicación es un prototipo académico basado en el dataset de Framingham. 
No sustituye un diagnóstico médico profesional.
""")

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["📊 Análisis Exploratorio", "🔮 Realizar Predicción", "📈 Métricas del Modelo"])

# --- Pestaña 1: EDA ---
with tab1:
    st.header("Análisis Exploratorio de Datos (EDA)")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribución de la Variable Objetivo")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df, x="TenYearCHD", ax=ax1)
            ax1.set_title("Casos de Enfermedad Coronaria (0=No, 1=Sí)")
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Presión Arterial Sistólica vs Riesgo")
            fig2, ax2 = plt.subplots()
            sns.boxplot(data=df, x="TenYearCHD", y="sysBP", ax=ax2)
            ax2.set_title("Presión Sistólica según Riesgo")
            st.pyplot(fig2)
        
        st.subheader("Mapa de Correlaciones")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm", center=0, ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("No se encontró el archivo 'framingham.csv' para mostrar los gráficos del EDA. Súbelo al repositorio.")

# --- Pestaña 2: Predicción ---
with tab2:
    st.header("Ingresa los datos del paciente")
    
    if modelo is None:
        st.error("⚠️ El modelo no está cargado. Asegúrate de que el archivo 'model/pipeline_framingham.pkl' esté en el repositorio.")
    else:
        # Crear columnas para el formulario
        col1, col2, col3 = st.columns(3)
        
        with col1:
            male = st.selectbox("Sexo Biológico", options=[1, 0], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
            age = st.number_input("Edad (años)", min_value=30, max_value=80, value=50)
            education = st.selectbox("Nivel Educativo (1-4)", options=[1.0, 2.0, 3.0, 4.0])
            currentSmoker = st.selectbox("¿Fuma actualmente?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
            cigsPerDay = st.number_input("Cigarrillos por día", min_value=0.0, max_value=70.0, value=0.0)
        
        with col2:
            BPMeds = st.selectbox("¿Toma medicamentos para la presión?", options=[1.0, 0.0], format_func=lambda x: "Sí" if x == 1.0 else "No")
            prevalentStroke = st.selectbox("¿Ha tenido un derrame cerebral?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
            prevalentHyp = st.selectbox("¿Tiene hipertensión?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
            diabetes = st.selectbox("¿Tiene diabetes?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
            totChol = st.number_input("Colesterol Total (mg/dL)", min_value=100.0, max_value=700.0, value=200.0)
        
        with col3:
            sysBP = st.number_input("Presión Arterial Sistólica (mmHg)", min_value=80.0, max_value=300.0, value=120.0)
            diaBP = st.number_input("Presión Arterial Diastólica (mmHg)", min_value=40.0, max_value=150.0, value=80.0)
            BMI = st.number_input("Índice de Masa Corporal (IMC)", min_value=15.0, max_value=60.0, value=25.0)
            heartRate = st.number_input("Frecuencia Cardíaca (lpm)", min_value=40.0, max_value=150.0, value=75.0)
            glucose = st.number_input("Glucosa (mg/dL)", min_value=40.0, max_value=400.0, value=80.0)

        # Botón de predicción
        if st.button("Calcular Riesgo"):
            # Crear DataFrame con los datos del formulario (mismo orden y nombres que en el entrenamiento)
            datos_paciente = pd.DataFrame([{
                'male': male, 'age': age, 'education': education, 'currentSmoker': currentSmoker,
                'cigsPerDay': cigsPerDay, 'BPMeds': BPMeds, 'prevalentStroke': prevalentStroke,
                'prevalentHyp': prevalentHyp, 'diabetes': diabetes, 'totChol': totChol,
                'sysBP': sysBP, 'diaBP': diaBP, 'BMI': BMI, 'heartRate': heartRate, 'glucose': glucose
            }])
            
            # Realizar predicción
            prediccion = modelo.predict(datos_paciente)[0]
            probabilidad = modelo.predict_proba(datos_paciente)[0][1]
            
            st.markdown("---")
            if prediccion == 1:
                st.error(f"**Alerta de Riesgo:** El modelo predice una **ALTA** probabilidad de enfermedad coronaria a 10 años.")
                st.metric(label="Probabilidad estimada", value=f"{probabilidad*100:.1f}%")
            else:
                st.success(f"**Riesgo Bajo:** El modelo predice una **BAJA** probabilidad de enfermedad coronaria a 10 años.")
                st.metric(label="Probabilidad estimada", value=f"{probabilidad*100:.1f}%")
            
            st.info("Recuerda: Este es un modelo académico y no reemplaza la opinión de un médico.")

# --- Pestaña 3: Métricas ---
with tab3:
    st.header("Rendimiento del Modelo")
    st.markdown("""
    El modelo seleccionado fue el **Árbol de Decisión**, que mostró un mejor equilibrio (F1-score) para detectar los casos positivos (clase 1) en comparación con la Regresión Logística.
    """)
    
    # Datos de métricas extraídos de tu notebook
    metricas_df = pd.DataFrame({
        "Métrica": ["Accuracy", "Precisión (Clase 1)", "Recall (Clase 1)", "F1-Score (Clase 1)"],
        "Valor": [0.7630, 0.2188, 0.2171, 0.2179]
    })
    
    st.table(metricas_df)
    
    st.subheader("Matriz de Confusión (Árbol de Decisión)")
    # Matriz de confusión extraída de tu notebook: TN=619, FP=100, FN=101, TP=28
    cm = np.array([[619, 100], [101, 28]])
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=['Predijo 0', 'Predijo 1'], yticklabels=['Real 0', 'Real 1'])
    ax_cm.set_ylabel('Realidad')
    ax_cm.set_xlabel('Predicción')
    st.pyplot(fig_cm)
    
    st.markdown("""
    **Limitaciones:**
    - El dataset está desbalanceado (mayoría de casos negativos).
    - El modelo tiene un alto número de Falsos Negativos (101), lo que significa que no detecta todos los casos de riesgo.
    - Es un prototipo académico.
    """)
