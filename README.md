# Accidentes de tránsito con víctimas fatales en Cartagena

Quise saber si, con los datos de accidentes de Cartagena, se puede predecir si un accidente con víctimas va a dejar al menos un muerto. Solo uso lo que se sabe al momento del reporte: fecha, hora, tipo de accidente, barrio, coordenadas y similares.

Autor: Jose Becerra

URL de la aplicación: https://becerrajose037-accidentes-cartagena-streamlit-app-lj2hel.streamlit.app

Repositorio en GitHub: https://github.com/Becerrajose037/accidentes-cartagena-streamlit

Este repositorio tiene el trabajo para la materia Aprendizaje de Máquinas: el código de limpieza y entrenamiento, el modelo guardado, la aplicación de Streamlit y el cuaderno con el proceso.

## Los datos

Los datos vienen de [Datos Abiertos Colombia](https://www.datos.gov.co/d/5z62-bn99), conjunto con id `5z62-bn99`, publicado por la Alcaldía de Cartagena. El CSV tiene una fila por cada persona o registro, así que un mismo accidente aparece varias veces. En la limpieza agrupo las filas por `ipat` (el número del accidente) y me quedo con una fila por accidente.

La variable que quiero predecir, `fatal`, vale 1 si el accidente tuvo al menos un muerto. Los muertos y los heridos no entran como variables del modelo, porque sería usar la respuesta para predecir la respuesta. Es un problema desbalanceado, así que la exactitud no dice mucho. Por eso comparo los modelos con el F1 de la clase fatal.

## Qué hay en la carpeta

| Archivo | Para qué sirve |
|---|---|
| `accidentes_cartagena.csv` | Datos crudos descargados del portal |
| `prepro.py` | Limpieza y creación de variables; la app lo necesita para cargar el modelo |
| `entrenar_modelo.py` | Compara modelos, ajusta hiperparámetros, evalúa y guarda el modelo |
| `modelo_accidentes.joblib` | Pipeline entrenado; recibe el CSV crudo |
| `resultados.json` | Métricas, hiperparámetros y mejor modelo |
| `app.py` | La aplicación de Streamlit |
| `requirements.txt` | Librerías que necesita la aplicación |
| `requirements-entrenamiento.txt` | Librerías para reentrenar, incluidas las de gráficas y pruebas estadísticas |
| `crisp_dm_accidentes.ipynb` | Cuaderno con el proceso, siguiendo CRISP-DM |
| `generar_notebook.py` | Genera el cuaderno a partir de `entrenar_modelo.py` |
| `figuras/` | Gráficas del informe, que también se muestran en la app |

## Correrla en mi computador

Uso Python 3.12, que es la versión con la que probé la app. Desde la carpeta del proyecto:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

En Linux o macOS la activación es `source venv/bin/activate`. Se abre el navegador en `http://localhost:8501`. La app tiene tres pestañas: predecir un accidente escribiendo sus datos, subir un CSV con varios accidentes para predecirlos todos, y "Acerca del modelo", con las métricas.

## Reentrenar el modelo

Para reentrenar hace falta instalar también las librerías de entrenamiento:

```bash
pip install -r requirements-entrenamiento.txt
python entrenar_modelo.py
```

El script consolida el CSV a una fila por accidente y crea la variable `fatal`. Separa 70% para entrenar y 30% para probar, de forma estratificada. Compara varios modelos (regresión logística, árbol, KNN, Naive Bayes, SVM y ensambles) con validación cruzada de 5 partes sobre el 70%, y ajusta hiperparámetros también solo con ese 70%. Las clases se balancean con SMOTE dentro del Pipeline, para que no se filtre información del conjunto de prueba. Al final evalúa el mejor modelo en el 30% de prueba y guarda el Pipeline completo en `modelo_accidentes.joblib`.

Ojo: al correrlo se sobrescriben `modelo_accidentes.joblib` y `resultados.json`.

El cuaderno de Colab hace lo mismo, por si prefiero correrlo ahí. Instala las librerías con versiones fijadas, descarga el CSV desde datos.gov.co si no está en el directorio y guarda `resultados.json`, las figuras de `figuras/` y el modelo como `modelo_colab.joblib`. Ese modelo no es el que usa la app: al contener `preparar` definida en el cuaderno (`__main__`), `app.py` no puede cargarlo. El archivo `modelo_accidentes.joblib` que carga la app se genera con `python entrenar_modelo.py`. Notebook de Colab: [pendiente]

Si cambio `entrenar_modelo.py` o `prepro.py`, regenero el cuaderno con `python generar_notebook.py` para que sigan siendo coherentes.

Las métricas del mejor modelo están en `resultados.json`, y también en la pestaña "Acerca del modelo" de la app. Ahí está el F1 de la clase fatal.

## Desplegarla en Streamlit Community Cloud

1. El repositorio tiene que ser público en GitHub, con `app.py`, `prepro.py`, `modelo_accidentes.joblib`, `accidentes_cartagena.csv`, `resultados.json`, `requirements.txt` y la carpeta `figuras/` (si falta, la app funciona pero no muestra las gráficas).
2. Entro a [share.streamlit.io](https://share.streamlit.io) con mi cuenta de GitHub y elijo "Create app".
3. En "Main file path" escribo `app.py`.
4. Si Cloud me pide la versión de Python en "Advanced settings", elijo 3.12.
5. Hago clic en "Deploy". Cuando termina, copio la URL y la pego arriba, donde dice `[pendiente]`.

## Limitaciones

- El modelo sirve solo para accidentes con víctimas. El dataset no incluye accidentes con solo daños, así que no dice nada sobre ellos.
- Los datos son de un solo municipio, Cartagena, entre 2020 y 2025.
- El modelo no se sostiene en 2025. En ese año el 63,5 % de los accidentes tienen al menos un muerto, mientras que entre 2020 y 2024 la proporción estuvo entre 5,8 % y 9,0 %. Cuando entrené con 2020 a 2024 y probé con 2025, el F1 de la clase fatal bajó a 0,073, el recall a 0,039 y el ROC AUC a 0,572. Eso quiere decir que el F1 de 0,745 del test aleatorio está inflado por ese cambio y no sirve para estimar lo que pasa en 2025. No sé si fue un cambio real en los accidentes o un cambio en la forma de registrarlos, así que tengo que confirmarlo con la fuente (Datos Abiertos Colombia) antes de confiar en el modelo para cualquier año reciente.
- La columna de gravedad es inconsistente en el CSV, así que la variable objetivo la construí con el conteo de muertos.
- Los códigos de hipótesis vienen del informe original y no tienen descripción en el CSV.
- El umbral de 0.5 es el que trae scikit-learn por defecto. No lo optimicé.
- El recall es modesto: el modelo deja pasar una parte de los accidentes fatales reales.
- Es un modelo académico. No debe usarse para decidir sobre tránsito, seguridad vial ni atención de emergencias.
