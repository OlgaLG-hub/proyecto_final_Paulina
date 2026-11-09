# ==============================================================================
# DICCIONARIO RÁPIDO DE ABREVIATURAS USADAS EN ESTE ARCHIVO:
# - st           = Streamlit (librería para crear la página web)
# - pd           = Pandas (librería para manejar tablas de datos)
# - plt          = Pyplot (módulo de dibujo dentro de la librería Matplotlib)
# - sklearn      = Scikit-Learn (librería de Inteligencia Artificial)
# - ia           = Inteligencia Artificial
# - def          = Define (definir o crear una función nueva)
# - csv          = Comma-Separated Values (archivo de tabla separado por comas, tipo Excel)
# - df           = DataFrame (estructura de tabla con filas y columnas en Pandas)
# - X            = Matriz de variables independientes (las columnas que sirven de pista)
# - y            = Vector de variable dependiente (la columna objetivo que queremos adivinar)
# - bus_type     = Business Type (tipo de negocio)
# - food_cat     = Food Category (categoría de alimento)
# - purchase_qty = Purchase Quantity (cantidad comprada)
# - storage_temp = Storage Temperature (temperatura de almacenamiento)
# - min / max    = Minimum / Maximum (valor mínimo / valor máximo)
# - col1 / col2  = Column 1 / Column 2 (columna izquierda / columna derecha)
# - tab1 / tab2  = Tab 1 / Tab 2 (pestaña 1 / pestaña 2)
# - fig1 / fig2  = Figure 1 / Figure 2 (figura o lienzo en blanco 1 y 2)
# - ax1 / ax2    = Axes 1 / Axes 2 (ejes o recuadro del gráfico 1 y 2)
# - fmt          = Format (formato visual de los números)
# - xlim / ylim  = X-limits / Y-limits (límites del eje horizontal X y del eje vertical Y)
# - xlabel/ylabel= X-label / Y-label (etiqueta de texto del eje X y del eje Y)
# - hist         = Histogram (gráfico de histograma)
# - axvline      = Axes Vertical Line (línea vertical dibujada sobre los ejes)
# - elif         = Else If ("si no, si...", para encadenar condiciones)
# - f"..."       = Formatted string (texto con formato que permite incrustar variables con {})
# - .1f / .2f    = Float (número con 1 o 2 decimales)
# ==============================================================================


# ==============================================================================
# 1. IMPORTAR LIBRERÍAS (HERRAMIENTAS DE PYTHON)
# ==============================================================================

# 'import ... as st': importamos Streamlit y lo abreviamos como 'st' (de ST-reamlit)
import streamlit as st

# 'import ... as pd': importamos Pandas y lo abreviamos como 'pd' (de Pan-Das)
import pandas as pd

# 'import ... as plt': importamos Pyplot de Matplotlib y lo abreviamos como 'plt' (de PLo-T, que significa gráfico)
import matplotlib.pyplot as plt

# 'sklearn' es la abreviatura de Scikit-Learn (librería de aprendizaje automático)
# Importamos 'RandomForestRegressor' (Bosque Aleatorio Regresor, algoritmo para predecir números)
from sklearn.ensemble import RandomForestRegressor

# Importamos 'OneHotEncoder' (Codificador One-Hot, traduce palabras como "Restaurant" a números 1 y 0)
from sklearn.preprocessing import OneHotEncoder

# Importamos 'ColumnTransformer' (Transformador de Columnas, aplica el traductor a las columnas elegidas)
from sklearn.compose import ColumnTransformer

# Importamos 'Pipeline' (Tubería, une el traductor y el modelo de IA en un solo paso)
from sklearn.pipeline import Pipeline

# Importamos las herramientas para separar los datos y evaluar la calidad del modelo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




# ==============================================================================
# 2. CONFIGURAR LA PÁGINA WEB CON 'st' (STREAMLIT)
# ==============================================================================

# 'st' = Streamlit | 'set_page_config' = Set Page Configuration (establecer configuración de página)
# 'layout="wide"' = diseño ancho para aprovechar toda la pantalla
st.set_page_config(page_title="El Rescatista de Alimentos", page_icon="🥗", layout="wide")

# 'st.title' = título principal grande en la web
st.title("🥗 El Rescatista de Alimentos - Predicción con IA")

# 'st.write' = escribir un texto normal en la página web
st.write("Simulador interactivo para predecir y reducir el desperdicio de comida en hostelería.")


# ==============================================================================
# 3. CARGAR DATOS Y ENTRENAR EL MODELO DE IA (INTELIGENCIA ARTIFICIAL)
# ==============================================================================

# '@st.cache_resource': 'st' = Streamlit | 'cache_resource' = guardar recurso en memoria caché (memoria rápida)
# Sirve para que el modelo se entrene una sola vez al abrir la web y no se repita en cada clic
@st.cache_resource
# 'def' = Define (define o crea una función llamada 'cargar_y_entrenar')
def cargar_y_entrenar():
    # 'pd' = Pandas | 'read_csv' = leer archivo CSV (Comma-Separated Values / Valores Separados por Comas)
    datos = pd.read_csv("food_waste_dataset.csv")

    # Lista con las 6 columnas que sirven de pista para la predicción
    columnas_pistas = [
        "Business_Type",        # Tipo de negocio
        "Food_Category",        # Categoría de comida
        "Purchase_Quantity",    # Cantidad comprada
        "Storage_Temperature",  # Temperatura de almacenamiento
        "Promotion",            # Promoción activa
        "Weather"               # Clima
    ]

    # 'X' (en mayúscula por convención matemática): guarda la matriz de pistas (las 6 columnas anteriores)
    X = datos[columnas_pistas]

    # 'y' (en minúscula por convención matemática): guarda la columna objetivo a adivinar (kilos desperdiciados)
    y = datos["Waste_Quantity"]

    # Lista con las columnas que contienen palabras (texto) en vez de números
    columnas_texto = ["Business_Type", "Food_Category", "Promotion", "Weather"]

    # Configuramos el traductor de palabras a números:
    # 'handle_unknown="ignore"' = ignorar palabras desconocidas sin dar error
    # 'remainder="passthrough"' = el resto de columnas (las numéricas) déjalas pasar tal cual
    traductor = ColumnTransformer(
        transformers=[("texto", OneHotEncoder(handle_unknown="ignore"), columnas_texto)],
        remainder="passthrough"
    )

    # 'modelo_ia' = Modelo de Inteligencia Artificial
    # 'n_estimators=50' = Number of Estimators (número de estimadores: 50 árboles de decisión)
    # 'random_state=42' = Estado aleatorio fijo (número semilla para que el entrenamiento dé el mismo resultado al repetirlo)
    modelo_ia = Pipeline(steps=[
        ("preprocesador", traductor),
        ("algoritmo", RandomForestRegressor(n_estimators=50, random_state=42))
    ])

    # Separamos los datos en un conjunto de entrenamiento y otro de prueba
    # 'test_size=0.2' = reserva el 20% de los datos para evaluar el modelo
    # 'random_state=42' = mantiene la misma división en cada ejecución
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Creamos un modelo independiente para medir su rendimiento con datos de prueba
    modelo_evaluacion = Pipeline(steps=[
        ("preprocesador", ColumnTransformer(
            transformers=[("texto", OneHotEncoder(handle_unknown="ignore"), columnas_texto)],
            remainder="passthrough"
        )),
        ("algoritmo", RandomForestRegressor(n_estimators=50, random_state=42))
    ])

    # Entrenamos el modelo de evaluación solamente con los datos de entrenamiento
    modelo_evaluacion.fit(X_train, y_train)

    # Generamos predicciones para los datos que el modelo no ha visto durante el entrenamiento
    predicciones_test = modelo_evaluacion.predict(X_test)

    # Calculamos el MAE: error absoluto medio en las mismas unidades que la variable objetivo
    mae = mean_absolute_error(y_test, predicciones_test)

    # Calculamos el RMSE: raíz del error cuadrático medio, que penaliza más los errores grandes
    rmse = mean_squared_error(y_test, predicciones_test) ** 0.5

    # Calculamos R²: proporción de la variabilidad de los datos explicada por el modelo
    r2 = r2_score(y_test, predicciones_test)

    # Entrenamos el modelo definitivo con todos los datos disponibles para realizar predicciones
    modelo_ia.fit(X, y)

    # 'return' = devolvemos los datos, el modelo entrenado y sus tres métricas de evaluación
    return datos, modelo_ia, mae, rmse, r2

# Ejecutamos la función:
# 'df' = DataFrame (Marco de Datos / Tabla completa leída del CSV)
# 'modelo' = la Inteligencia Artificial ya entrenada
# 'mae', 'rmse' y 'r2' = métricas que indican el rendimiento del modelo
df, modelo, mae, rmse, r2 = cargar_y_entrenar()


# ==============================================================================
# 4. MENÚ LATERAL IZQUIERDO ('st.sidebar') CON LAS OPCIONES DEL USUARIO
# ==============================================================================

# 'st' = Streamlit | 'sidebar' = barra lateral izquierda | 'header' = encabezado o título
st.sidebar.header("⚙️ Parámetros del Negocio")

# 'bus_type' = abreviatura de Business Type (Tipo de Negocio)
# 'selectbox' = caja de selección (menú desplegable)
# 'df["Business_Type"].unique()' = saca de la tabla 'df' los valores únicos (sin repetir)
bus_type = st.sidebar.selectbox("Tipo de Negocio", df["Business_Type"].unique())

# 'food_cat' = abreviatura de Food Category (Categoría de Alimento)
food_cat = st.sidebar.selectbox("Categoría de Alimento", df["Food_Category"].unique())

# 'purchase_qty' = abreviatura de Purchase Quantity (Cantidad Comprada)
# 'slider' = barra deslizante | 'min_value' = valor mínimo (10) | 'max_value' = valor máximo (500) | 'value' = valor inicial (100)
purchase_qty = st.sidebar.slider("Cantidad Comprada (kg)", min_value=10, max_value=500, value=100)

# 'storage_temp' = abreviatura de Storage Temperature (Temperatura de Almacenamiento en °C = grados centígrados)
storage_temp = st.sidebar.slider("Temperatura de Almacenamiento (°C)", min_value=-5.0, max_value=25.0, value=4.0)

# 'promotion' = promoción activa (Yes / No)
promotion = st.sidebar.selectbox("¿Hay Promoción?", df["Promotion"].unique())

# 'weather' = clima del día
weather = st.sidebar.selectbox("Clima", df["Weather"].unique())


# ==============================================================================
# 5. GUARDAR LA SELECCIÓN DEL USUARIO Y CALCULAR LA PREDICCIÓN
# ==============================================================================

# Agrupamos en un diccionario las 6 variables abreviadas que acabamos de recoger en la barra lateral
datos_elegidos = {
    "Business_Type": [bus_type],
    "Food_Category": [food_cat],
    "Purchase_Quantity": [purchase_qty],
    "Storage_Temperature": [storage_temp],
    "Promotion": [promotion],
    "Weather": [weather]
}

# 'pd.DataFrame' = usamos Pandas (pd) para crear una tabla de 1 fila ('DataFrame') con la selección del usuario
entrada_usuario = pd.DataFrame(datos_elegidos)

# '.predict(...)[0]' = predecir el desperdicio para esa fila y extraer el primer número con '[0]'
prediccion_actual = modelo.predict(entrada_usuario)[0]

# Si el número predicho es menor que 0, lo fijamos en 0.0
if prediccion_actual < 0:
    prediccion_actual = 0.0

# Calculamos el porcentaje dividiendo la predicción entre 'purchase_qty' (Purchase Quantity / Cantidad Comprada) y multiplicando por 100
porcentaje_desperdicio = (prediccion_actual / purchase_qty) * 100


# ==============================================================================
# EVALUACIÓN DE LA PRECISIÓN DEL MODELO
# ==============================================================================

# Mostramos las métricas calculadas con el conjunto de datos de prueba
st.subheader("🎯 Evaluación de la precisión del modelo")

# Dividimos el espacio en tres columnas para mostrar cada métrica por separado
col_mae, col_rmse, col_r2 = st.columns(3)

# MAE = error absoluto medio, expresado en kilogramos; cuanto menor, mejor
with col_mae:
    st.metric("MAE (Error medio)", f"{mae:.2f} kg")

# RMSE = raíz del error cuadrático medio; cuanto menor, mejor
with col_rmse:
    st.metric("RMSE (Error cuadrático)", f"{rmse:.2f} kg")

# R² = proporción de variabilidad explicada por el modelo; cuanto más cerca de 1, mejor
with col_r2:
    st.metric("R² (Coeficiente de determinación)", f"{r2:.2f}")

# Explicamos brevemente cómo interpretar los tres indicadores
st.caption(
    "MAE y RMSE: cuanto menores sean, menor es el error de predicción. "
    "R²: cuanto más cerca de 1, mejor explica el modelo la variabilidad de los datos. "
    "Estas métricas se calculan con datos de prueba que no se utilizaron para entrenar el modelo."
)


# ==============================================================================
# 6. RESUMEN DE LA CONSULTA Y DIVIDIR LA PANTALLA EN DOS COLUMNAS
# ==============================================================================

# ------------------------------------------------------------------------------
# RESUMEN DE LA CONSULTA (ANCHO COMPLETO)
# ------------------------------------------------------------------------------
st.subheader("📊 Resumen de la Consulta")

# 'hide_index=True' = oculta el índice de la tabla (la columna donde aparece el 0)
st.dataframe(entrada_usuario, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# DIVIDIR EL RESTO DE LA PANTALLA EN 'col1' (COLUMNA 1) Y 'col2' (COLUMNA 2)
# ------------------------------------------------------------------------------

# 'col1, col2' = abreviaturas de Column 1 (columna izquierda) y Column 2 (columna derecha)
# 'st.columns(2)' = orden de Streamlit (st) para dividir la pantalla en 2 columnas iguales
col1, col2 = st.columns(2)


# ------------------------------------------------------------------------------
# COLUMNA IZQUIERDA ('with col1:' = dentro de la Columna 1)
# ------------------------------------------------------------------------------
with col1:
    # 'st.button' = dibuja un botón en la web | 'type="primary"' = estilo principal (color rojo destacado)
    st.button("🔮 Calcular Desperdicio Estimado", type="primary", use_container_width=True)

    # 'st.metric' = muestra un indicador numérico grande en pantalla
    # 'f"{prediccion_actual:.2f} kg"':
    #   - 'f' = Formatted string (texto con formato que sustituye lo que va entre llaves {} por su valor)
    #   - ':.2f' = Float con 2 decimales (redondea el número a 2 decimales)
    #   - 'kg' = Kilogramos
    # ':.1f' = Float con 1 decimal (redondea el porcentaje a 1 decimal)
    # 'delta_color="inverse"' = invierte los colores del indicador (rojo para subidas, verde para bajadas)
    st.metric(
        label="Desperdicio Estimado",
        value=f"{prediccion_actual:.2f} kg",
        delta=f"{porcentaje_desperdicio:.1f}% de la compra",
        delta_color="inverse"
    )

    # Subtítulo para las recomendaciones
    st.subheader("💡 Recomendaciones del Asistente")

    # 'if' = si se cumple la condición (> 10 kg)
    # 'elif' = abreviatura de Else If ("si no, si se cumple esta otra condición" > 4 kg)
    # 'else' = en cualquier otro caso (4 kg o menos)
    if prediccion_actual > 10:
        # 'st.error' = caja roja de alerta en Streamlit
        st.error("🚨 ALERTA: Desperdicio Alto. Reduce el pedido de compra o revisa la temperatura de la cámara.")
    elif prediccion_actual > 4:
        # 'st.warning' = caja amarilla/naranja de advertencia en Streamlit
        st.warning("⚠️ AVISO: Desperdicio Moderado. Puedes activar una promoción o donar el excedente.")
    else:
        # 'st.success' = caja verde de éxito en Streamlit
        st.success("✅ ÓPTIMO: Desperdicio bajo control. El volumen de compra y las condiciones son muy eficientes.")


# ------------------------------------------------------------------------------
# COLUMNA DERECHA ('with col2:' = dentro de la Columna 2)
# ------------------------------------------------------------------------------
with col2:
    # Subtítulo de la columna derecha
    st.subheader("📈 Análisis Dinámico en Tiempo Real")

    # 'tab1, tab2' = abreviaturas de Tab 1 (Pestaña 1) y Tab 2 (Pestaña 2)
    # 'st.tabs' = crea pestañas clicables en Streamlit
    tab1, tab2 = st.tabs(["📊 Impacto por Negocio", "📉 Distribución y Tu Caso"])

    # Elegimos el color ('red', 'orange', 'green') usando 'if', 'elif' (Else If) y 'else'
    if prediccion_actual > 10:
        color_alerta = "red"      # Rojo para desperdicio alto
    elif prediccion_actual > 4:
        color_alerta = "orange"   # Naranja para desperdicio moderado
    else:
        color_alerta = "green"    # Verde para desperdicio bajo

    # === PESTAÑA 1 ('with tab1:' = dentro de la Pestaña 1) ===
    with tab1:
        # 'st.write' = escribe un texto explicativo en Streamlit
        st.write("**Comparativa: Tu predicción de hoy frente a la media de tu sector**")

        # Filtramos la tabla 'df' (DataFrame) donde la columna coincida con 'bus_type' (Business Type)
        datos_del_sector = df[df["Business_Type"] == bus_type]

        # '.mean()' = calcula la media (promedio matemático) de la columna 'Waste_Quantity'
        media_sector = datos_del_sector["Waste_Quantity"].mean()

        # Textos para el eje horizontal (usando 'f"..."' = Formatted string para incluir el nombre de 'bus_type')
        etiquetas = ["Tu Predicción", f"Media en {bus_type}"]

        # Valores numéricos para la altura de las dos barras
        valores = [prediccion_actual, media_sector]

        # 'fig1, ax1' = abreviaturas de Figure 1 (Figura/Lienzo 1) y Axes 1 (Ejes/Gráfico 1)
        # 'plt.subplots' = función de Pyplot (plt) que crea el lienzo y los ejes del gráfico
        # 'figsize=(5, 3.5)' = abreviatura de Figure Size (Tamaño de Figura: 5 de ancho por 3.5 de alto)
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))

        # 'ax1.bar' = dibuja un gráfico de barras dentro de los ejes 'ax1' (Axes 1)
        barras = ax1.bar(etiquetas, valores, color=[color_alerta, "gray"])

        # 'ax1.bar_label' = pone etiquetas de texto encima de las barras en 'ax1' (Axes 1)
        # 'fmt="%.1f kg"' = abreviatura de Format (Formato: número decimal Float con 1 decimal seguido de "kg")
        # 'padding=3' = separación de 3 puntos entre la barra y el número
        ax1.bar_label(barras, fmt="%.1f kg", padding=3)

        # 'ax1.set_ylabel' = Set Y-Label (establecer la etiqueta o título del eje vertical Y en 'ax1')
        ax1.set_ylabel("Kilogramos (kg)")

        # 'ax1.set_ylim' = Set Y-Limits (establecer los límites mínimo y máximo del eje vertical Y en 'ax1')
        # 'max(valores) + 5' = calcula el máximo de los dos valores y le suma 5 kg de margen superior
        ax1.set_ylim(0, max(valores) + 5)

        # 'st.pyplot(fig1)' = Streamlit (st) dibuja el gráfico de Python ('pyplot' = Python Plot) contenido en 'fig1' (Figure 1)
        st.pyplot(fig1)

    # === PESTAÑA 2 ('with tab2:' = dentro de la Pestaña 2) ===
    with tab2:
        # Texto explicativo encima del segundo gráfico
        st.write("**En qué punto se sitúa tu predicción respecto a todos los casos históricos**")

        # 'fig2, ax2' = Figure 2 (Figura/Lienzo 2) y Axes 2 (Ejes/Gráfico 2)
        # 'plt.subplots' = crea con Pyplot (plt) el segundo gráfico con 'figsize' (Figure Size / Tamaño de Figura) de 5x3.5
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))

        # 'ax2.hist' = dibuja en 'ax2' (Axes 2) un 'hist' (abreviatura de Histogram / Histograma)
        # 'df["Waste_Quantity"]' = saca la columna de desperdicio de la tabla 'df' (DataFrame)
        # 'bins=40' = divide el histograma en 40 contenedores o barritas verticales
        ax2.hist(df["Waste_Quantity"], bins=40, color="lightgray", edgecolor="white", label="Historial")

        # 'ax2.axvline' = abreviatura de Axes Vertical Line (dibuja una Línea Vertical en los Ejes 'ax2')
        # 'x=prediccion_actual' = posición en el eje horizontal X donde se dibuja la línea
        # 'linestyle="--"' = estilo de línea discontinua (a rayas)
        # 'linewidth=3' = grosor de la línea de 3 puntos
        # 'f"Tu caso: {prediccion_actual:.1f} kg"' = Formatted string (f) con 1 decimal (.1f) en kilogramos (kg)
        ax2.axvline(
            x=prediccion_actual, 
            color=color_alerta, 
            linestyle="--", 
            linewidth=3, 
            label=f"Tu caso: {prediccion_actual:.1f} kg"
        )

        # 'ax2.set_xlim(0, 50)' = Set X-Limits (establecer los límites del eje horizontal X entre 0 y 50 kg para hacer zoom)
        ax2.set_xlim(0, 50)

        # 'ax2.set_xlabel' = Set X-Label (establecer el título del eje horizontal X en 'ax2')
        ax2.set_xlabel("Kilogramos de Desperdicio")

        # 'ax2.set_ylabel' = Set Y-Label (establecer el título del eje vertical Y en 'ax2')
        ax2.set_ylabel("Cantidad de Días / Registros")

        # 'ax2.legend()' = muestra el recuadro de leyenda en 'ax2' (Axes 2)
        ax2.legend()

        # 'st.pyplot(fig2)' = Streamlit (st) muestra en la web el gráfico ('pyplot' = Python Plot) guardado en 'fig2' (Figure 2)
        st.pyplot(fig2)



# El modelo de regresión Random Forest ha obtenido los siguientes resultados:
#
# MAE = 3,03 kg: el error absoluto medio de las predicciones es de aproximadamente 3,03 kg.
#
# RMSE = 6,28 kg: este indicador refleja los errores de predicción y penaliza especialmente los errores más grandes.
#
# R² = 0,87: el modelo explica aproximadamente el 87 % de la variabilidad de los datos de desperdicio alimentario.
#
# Conclusión:
#
# Los resultados indican que el modelo presenta un buen rendimiento predictivo. El valor de R² demuestra que consigue explicar una gran parte de la variación del desperdicio alimentario. Sin embargo, todavía existe margen de mejora, especialmente para reducir los errores más elevados.