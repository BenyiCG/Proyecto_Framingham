# Proyecto Final - Minería de Datos (Framingham)

**Estudiante:** Benjamín José Correa Guerra  
**Asignatura:** Minería de Datos  
**Universidad:** Universidad de la Costa  

## Descripción del Proyecto
Aplicación web interactiva desarrollada con Streamlit para predecir el riesgo de enfermedad coronaria a 10 años, basada en el histórico dataset de Framingham. El proyecto implementa un pipeline de `scikit-learn` (imputación, escalamiento y codificación One-Hot) y un modelo de Regresión Logística con balanceo de clases (`class_weight='balanced'`) para priorizar la detección de pacientes enfermos (Recall del 60%).

## Enlaces
- **Aplicación en producción (Streamlit):** [https://proyectoframingham-yv3cnpgsfujl7zxhqlvwue.streamlit.app/](https://proyectoframingham-yv3cnpgsfujl7zxhqlvwue.streamlit.app/)
- **Notebook del proyecto:** `Proyecto_Final_Mineria_Datos_Framingham.ipynb`

## Cómo ejecutar el proyecto localmente
1. Clona el repositorio:
   ```bash
   git clone https://github.com/BenyiCG/Proyecto_Framingham.git
  
 Instala las dependencias:
 
   ```bash
pip install -r requirements.txt
Ejecuta la aplicación de Streamlit:

bash
streamlit run app.py
