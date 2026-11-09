📋 Resumen del Proyecto: "El Rescatista de Alimentos"
1. ¿Qué hace nuestra aplicación?
Es un simulador web interactivo creado con Streamlit que permite a distintos establecimientos de alimentación (Restaurant, Cafeteria, Cloud Kitchen y Supermarket) predecir cuántos kilogramos de comida van a desperdiciar según sus condiciones del día, dándoles avisos y recomendaciones para evitarlo antes de que ocurra.

2. ¿Cómo funciona por dentro (Datos e Inteligencia Artificial)?

Lectura de datos (Pandas): El programa abre el archivo histórico food_waste_dataset.csv.

Pistas y objetivo (X e y): El modelo estudia 6 variables como pistas (Business_Type, Food_Category, Purchase_Quantity, Storage_Temperature, Promotion y Weather) para aprender a calcular la columna objetivo: los kilos de merma (Waste_Quantity).

Traducción y entrenamiento (Scikit-Learn): Como el algoritmo matemático no lee palabras, usamos un traductor (ColumnTransformer + OneHotEncoder) que convierte los textos en números, conectado mediante una tubería (Pipeline) a un modelo RandomForestRegressor de 50 árboles de decisión.

Memoria rápida (@st.cache_resource): Guardamos el modelo entrenado en la caché de la web para que no tenga que reentrenarse cada vez que cambiamos un dato en la pantalla.

3. ¿Cómo se organiza la pantalla de la web?

Menú lateral izquierdo (st.sidebar): Contiene los desplegables y barras deslizantes donde el usuario introduce los datos de su local.

Columna izquierda (Resumen de la Consulta y Recomendaciones del Asistente): Muestra una tabla con las opciones elegidas y calcula el Desperdicio Estimado en kilogramos y en porcentaje sobre la compra. Debajo, aplica un sistema de semáforo que muestra una caja roja (ALERTA: Desperdicio Alto) si supera los 10 kg, naranja si es moderado (entre 4 y 10 kg), o verde (ÓPTIMO: Desperdicio Bajo control) si es de 4 kg o menos.

Columna derecha (Análisis Dinámico en Tiempo Real): Utiliza Matplotlib dividido en dos pestañas interactivas para comparar los resultados sin saturar la pantalla:

Pestaña 1 (Impacto por Negocio): Gráfico de barras que compara los kilos de nuestra predicción frente al desperdicio medio histórico de ese sector.

Pestaña 2 (Distribución y Tu Caso): Histograma con el historial de registros y una línea vertical discontinua que se mueve en vivo y cambia de color (verde, naranja o rojo) según nuestra predicción.


