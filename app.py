import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# 1. Configuración de la página web
st.set_page_config(
    page_title='El Rescatista de Alimentos', page_icon='🍲', layout='wide'
)


# 2. Cargar datos y entrenar el modelo automáticamente
@st.cache_resource
def cargar_datos_y_entrenar():
  # Carga el dataset que subiste a la carpeta
  df = pd.read_csv('food_waste_dataset.csv')

  # Limpiamos valores nulos básicos para el modelo
  df_clean = df.dropna(
      subset=[
          'Business_Type',
          'Food_Category',
          'Purchase_Quantity',
          'Storage_Temperature',
          'Promotion',
          'Weather',
          'Waste_Quantity',
      ]
  ).copy()

  # Variables predictoras (X) y objetivo (y)
  features = [
      'Business_Type',
      'Food_Category',
      'Purchase_Quantity',
      'Storage_Temperature',
      'Promotion',
      'Weather',
  ]
  target = 'Waste_Quantity'

  X = df_clean[features]
  y = df_clean[target]

  # Preprocesador para transformar texto a categorías numéricas
  preprocessor = ColumnTransformer(
      transformers=[
          (
              'cat',
              OneHotEncoder(handle_unknown='ignore'),
              ['Business_Type', 'Food_Category', 'Promotion', 'Weather'],
          )
      ],
      remainder='passthrough',
  )

  # Pipeline con Random Forest
  modelo = Pipeline(
      steps=[
          ('preprocessor', preprocessor),
          (
              'regressor',
              RandomForestRegressor(n_estimators=50, random_state=42),
          ),
      ]
  )

  modelo.fit(X, y)
  return modelo, df_clean


modelo, df = cargar_datos_y_entrenar()

# 3. Interfaz Principal (Streamlit)
st.title('🍲 El Rescatista de Alimentos')
st.subheader(
    'Proyecto Final: Predicción de Desperdicio Alimentario con IA'
)  #[cite: 1, 3]
st.write(
    'Introduce los parámetros operativos y ambientales del establecimiento en'
    ' el panel izquierdo para estimar los kilogramos de desperdicio.'
)  #[cite: 3, 9]

# 4. Barra Lateral (Inputs del usuario)
st.sidebar.header('⚙️ Parámetros de Simulación')

bus_type = st.sidebar.selectbox(
    '🏢 Tipo de Negocio:', df['Business_Type'].unique()
)
food_cat = st.sidebar.selectbox(
    '🍎 Categoría de Alimento:', df['Food_Category'].unique()
)
purchase_qty = st.sidebar.slider(
    '📦 Cantidad Comprada (kg/unidades):', 10, 300, 100, 5
)
storage_temp = st.sidebar.slider(
    '🌡️ Temperatura de Almacenamiento (°C):', 0.0, 35.0, 15.0, 0.5
)
promotion = st.sidebar.selectbox('🏷️ ¿Hay Promoción activa?', ['No', 'Yes'])
weather = st.sidebar.selectbox(
    '🌤️ Clima / Tiempo:', ['Sunny', 'Cloudy', 'Rainy', 'Hot', 'Cold']
)

# Estructurar los datos de entrada para la predicción
entrada_usuario = pd.DataFrame({
    'Business_Type': [bus_type],
    'Food_Category': [food_cat],
    'Purchase_Quantity': [purchase_qty],
    'Storage_Temperature': [storage_temp],
    'Promotion': [promotion],
    'Weather': [weather],
})

# 5. Columnas de Resultados y Gráfica
col1, col2 = st.columns([1, 1])

with col1:
  st.markdown('### 📊 Resumen de la Consulta')
  st.dataframe(entrada_usuario, use_container_width=True)

  # Botón de predicción
  if st.button(
      '🔮 Calcular Desperdicio Estimado',
      type='primary',
      use_container_width=True,
  ):
    prediccion = modelo.predict(entrada_usuario)[0]
    prediccion = max(0.0, prediccion)  # Evitar negativos

    st.metric(
        label='Desperdicio Estimado',
        value=f'{prediccion:.2f} kg',
        delta=f'{(prediccion / purchase_qty * 100):.1f}% de la compra',
        delta_color='inverse',
    )

    # Reto Extra: Recomendaciones inteligentes según el resultado
    st.markdown('### 💡 Recomendaciones del Asistente')
    if prediccion > 10:
      st.error(
          '🚨 **ALERTA: Desperdicio Alto**\n\n'
          '- Revisar la temperatura de almacenamiento (`Storage_Temperature`).\n'
          '- Considerar activar ofertas de descuento si hay excedente'
          ' acumulado.\n'
          '- Planificar lotes de compra más pequeños para esta categoría.'
      )
    elif prediccion > 4:
      st.warning(
          '⚠️ **AVISO: Desperdicio Moderado**\n\n'
          '- Vigilar la rotación de stock de productos perecederos.\n'
          '- Evaluar el impacto de las promociones en la demanda real.'
      )
    else:
      st.success(
          '✅ **ÓPTIMO: Desperdicio Bajo control**\n\n'
          '- El volumen de compra y las condiciones actuales son muy eficientes.'
      )

with col2:
  st.markdown('### 📈 Análisis Histórico vs Tu Predicción')

  # Crear la figura para Matplotlib
  fig, ax = plt.subplots(figsize=(6, 4))

  # 1. Pintar todos los puntos históricos del dataset
  ax.scatter(
      df['Purchase_Quantity'],
      df['Waste_Quantity'],
      alpha=0.3,
      color='#4CAF50',
      label='Registros históricos',
  )

  # 2. CALCULAR LA PREDICCIÓN ACTUAL EN VIVO
  prediccion_actual = modelo.predict(entrada_usuario)[0]
  prediccion_actual = max(0.0, prediccion_actual)

  # 3. PINTAR EL PUNTO ROJO QUE SE MUEVE EN TIEMPO REAL
  ax.scatter(
      [purchase_qty],  # Valor de compra que el usuario mueve en la barra
      [prediccion_actual],  # Resultado que predice el modelo para hoy
      color='#D32F2F',  # Color rojo destacado
      s=150,  # Tamaño grande del punto
      zorder=5,  # Para que se dibuje por encima de los demás
      label='Tu selección de HOY',
  )

  # Configurar etiquetas y diseño del gráfico
  ax.set_xlabel('Cantidad Comprada (kg)')
  ax.set_ylabel('Desperdicio Estimado (kg)')
  ax.set_title('Relación Compra vs Desperdicio')
  ax.legend(fontsize=9)
  ax.grid(True, linestyle='--', alpha=0.5)

  # Mostrar el gráfico dentro de Streamlit
  st.pyplot(fig)
