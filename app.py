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
  st.markdown('### 📈 Análisis Dinámico en Tiempo Real')

  # Pestañas para alternar entre los dos gráficos interactivos
  tab1, tab2 = st.tabs(['📊 Impacto por Negocio', '📉 Distribución y Tu Caso'])

  with tab1:
    st.markdown(
        '**Media histórica vs. tu selección actual según el tipo de'
        ' negocio:**'
    )

    # Agrupamos los datos reales del dataset
    df_barras = (
        df.groupby('Business_Type')['Waste_Quantity'].mean().reset_index()
    )

    fig1, ax1 = plt.subplots(figsize=(5, 3.5))

    # Colores base para las barras
    colores = [
        '#81C784' if b != bus_type else '#D32F2F'
        for b in df_barras['Business_Type']
    ]

    ax1.bar(
        df_barras['Business_Type'],
        df_barras['Waste_Quantity'],
        color=colores,
        edgecolor='white',
    )
    ax1.set_ylabel('Desperdicio Medio (kg)')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Leyenda visual para explicar el color rojo
    ax1.text(
        0.05,
        0.85,
        '🔴 Tu negocio seleccionado',
        transform=ax1.transAxes,
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#D32F2F'),
    )

    st.pyplot(fig1)

 with tab2:
    st.markdown(
        '**Distribución del desperdicio y nivel de alerta para tu selección:**'
    )

    # 1. Calcular la predicción actual en vivo
    prediccion_actual = modelo.predict(entrada_usuario)[0]
    prediccion_actual = max(0.0, prediccion_actual)

    # 2. Definir el color dinámico según el nivel de desperdicio
    if prediccion_actual > 10:
      color_alerta = '#D32F2F'  # Rojo (Alerta alta)
      texto_alerta = 'Nivel Alto (Rojo)'
    elif prediccion_actual > 4:
      color_alerta = '#F57C00'  # Naranja / Amarillo (Aviso moderado)
      texto_alerta = 'Nivel Moderado (Naranja)'
    else:
      color_alerta = '#388E3C'  # Verde (Óptimo / Controlado)
      texto_alerta = 'Nivel Óptimo (Verde)'

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))

    # Histograma de fondo con todo el histórico
    ax2.hist(
        df['Waste_Quantity'],
        bins=40,
        color='#B0BEC5',
        edgecolor='white',
        alpha=0.6,
        label='Histórico general',
    )

    # Línea vertical dinámica que cambia de color según el estado
    ax2.axvline(
        x=prediccion_actual,
        color=color_alerta,
        linestyle='--',
        linewidth=3.0,
        label=f'Tu Predicción: {prediccion_actual:.1f} kg ({texto_alerta})',
    )

    # Hacemos zoom en el eje X para ver el detalle (hasta 60 kg)
    ax2.set_xlim(0, 60)

    ax2.set_xlabel('Kilogramos de Desperdicio')
    ax2.set_ylabel('Número de Registros')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    st.pyplot(fig2)
