import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from prepro import LAT_RANGO, LON_RANGO, preparar  # noqa: F401 (preparar hace falta para cargar el Pipeline)

BASE = Path(__file__).parent
RUTA_MODELO = BASE / 'modelo_accidentes.joblib'
RUTA_CSV = BASE / 'accidentes_cartagena.csv'
RUTA_RESULTADOS = BASE / 'resultados.json'
RUTA_FIGURAS = BASE / 'figuras'

CENTRO_CARTAGENA = (10.40, -75.50)
SIN_DATO = 'Sin dato'
COLUMNAS_CSV = [
    'ipat', 'fecha', 'hora', 'gravedad', 'clase_accidente', 'choque_con', 'objeto_fijo',
    'hipotesis_1', 'hipotesis_2', 'hipotesis_3', 'hipotesis_4', 'barrio', 'lugar',
    'via1', 'via2', 'via3', 'via4', 'via5', 'via6', 'localidad', 'muertos', 'heridos',
    'coordenadas', 'latitud', 'longitud',
]
COLUMNAS_NECESARIAS = [
    'fecha', 'hora', 'clase_accidente', 'choque_con', 'objeto_fijo', 'hipotesis_1',
    'hipotesis_2', 'barrio', 'localidad', 'via6', 'latitud', 'longitud',
]
AVISO = ('Este es un modelo académico hecho por un estudiante con datos públicos. '
         'No es una herramienta de decisión: no sirve para tomar decisiones de tránsito, '
         'seguridad vial ni de emergencias.')

st.set_page_config(page_title='Accidentes fatales en Cartagena', layout='wide')


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_resultados():
    with open(RUTA_RESULTADOS, encoding='utf-8') as f:
        return json.load(f)


def opciones_texto(serie):
    valores = serie.dropna().astype(str).str.strip().str.lower().str.capitalize()
    return sorted(valores.unique())


@st.cache_data
def opciones_formulario():
    datos = pd.read_csv(RUTA_CSV)
    fechas = pd.to_datetime(datos['fecha'], errors='coerce')
    return {
        'fecha_min': fechas.min().date(),
        'fecha_max': fechas.max().date(),
        'clases': opciones_texto(datos['clase_accidente']),
        'choque': [SIN_DATO] + opciones_texto(datos['choque_con']),
        'objeto': [SIN_DATO] + opciones_texto(datos['objeto_fijo']),
        'hipotesis': sorted(int(v) for v in datos['hipotesis_1'].dropna().unique()),
        'barrios': sorted(datos['barrio'].dropna().str.strip().str.upper().unique()),
        'localidades': sorted(int(v) for v in datos['localidad'].dropna().unique()),
        'vias6': opciones_texto(datos['via6']),
    }


def indice(opciones, valor):
    return opciones.index(valor) if valor in opciones else 0


def cifras_2025(res):
    tasas = res['tasa_fatal_por_anio']
    otros = [v['tasa_pct'] for anio, v in tasas.items() if anio != '2025']
    formato = lambda valor: f'{valor:.1f}'.replace('.', ',')
    return formato(tasas['2025']['tasa_pct']), formato(min(otros)), formato(max(otros))


def etiqueta_hipotesis(codigo):
    return 'Ninguna (0)' if codigo == 0 else f'Código {codigo}'


def etiqueta_via6(valor):
    return 'No aplica (0)' if valor == '0' else valor


def construir_fila(fecha, hora, clase, choque, objeto, hip1, hip2, barrio, localidad, via6, lat, lon):
    sin_dato = lambda v: None if v == SIN_DATO else v
    fila = {columna: None for columna in COLUMNAS_CSV}
    fila.update({
        'ipat': 0,
        'fecha': f'{fecha.isoformat()}T00:00:00.000',
        'hora': f'1899-12-31T{hora:02d}:00:00.000',
        'clase_accidente': clase,
        'choque_con': sin_dato(choque),
        'objeto_fijo': sin_dato(objeto),
        'hipotesis_1': hip1,
        'hipotesis_2': hip2,
        'hipotesis_3': 0,
        'hipotesis_4': 0,
        'barrio': barrio,
        'via1': '0', 'via2': '0', 'via3': '0', 'via4': '0', 'via5': '0',
        'via6': via6,
        'localidad': localidad,
        'muertos': 0,
        'heridos': 0,
        'coordenadas': f'{lat},{lon}',
        'latitud': lat,
        'longitud': lon,
    })
    return pd.DataFrame([fila], columns=COLUMNAS_CSV)


def pestana_predecir(modelo, opciones):
    st.warning(AVISO)
    st.write('Elige las características del accidente y calcula la probabilidad de que tenga al menos un muerto.')

    with st.expander('¿Qué significa cada campo?'):
        st.markdown(
            '- **Fecha del accidente:** el modelo mira el año, el mes, el día de la semana y si cae en fin de semana. '
            'El año pesa mucho (ver la sección "Acerca del modelo").\n'
            '- **Hora del accidente:** la hora en que ocurrió. Se usa también como franja: madrugada (0 a 5), '
            'mañana (6 a 11), tarde (12 a 17) y noche (18 a 23).\n'
            '- **Clase de accidente:** cómo ocurrió. *Choque* (dos cosas chocan), *Atropello* (un vehículo golpea a una persona), '
            '*Caída ocupante* (alguien cae de un vehículo), *Volcamiento* (el vehículo se voltea) u *Otro*.\n'
            '- **Choque con:** contra qué chocó el vehículo. *Vehículo*, *Objeto fijo*, *Semoviente* (un animal) o *No*. '
            '"Sin dato" significa que el informe no lo registró.\n'
            '- **Objeto fijo:** si el vehículo chocó contra algo fijo, como un poste, un muro o un árbol. '
            '"Ninguno" quiere decir que no hubo objeto fijo.\n'
            '- **Tipo de vía (vía 6):** categoría de una de las vías registradas en el accidente. '
            '"No aplica (0)" significa que el informe no tiene una sexta vía.\n'
            '- **Hipótesis 1 y 2:** códigos numéricos de las posibles causas del accidente, según la primera '
            'apreciación del agente de tránsito. El código de la hipótesis 2 puede ser 0, que significa "ninguna". '
            'La lista de códigos no está traducida en el dataset, así que aquí los muestro tal cual.\n'
            '- **Barrio:** el barrio donde ocurrió el accidente, tal como aparece en el registro.\n'
            '- **Localidad:** la localidad de Cartagena donde ocurrió (1, 2 o 3).\n'
            '- **Latitud y longitud:** la ubicación. Solo se aceptan valores dentro de Cartagena (latitud entre 10,1 y 10,6, '
            'longitud entre -75,8 y -75,2). Si no sabes la coordenada exacta, deja el centro.\n\n'
            'Después de calcular, el modelo muestra una probabilidad entre 0 % y 100 %. Si es 50 % o más, '
            'la predicción es "Fatal".'
        )

    with st.form('formulario_accidente'):
        col1, col2 = st.columns(2)
        with col1:
            fecha = st.date_input('Fecha del accidente', value=date(2024, 6, 15),
                                  min_value=opciones['fecha_min'], max_value=opciones['fecha_max'],
                                  help='Año, mes y día del accidente. Solo se aceptan fechas entre 2020 y 2025.')
            hora = st.slider('Hora del accidente', 0, 23, 17,
                             help='Hora en que ocurrió, de 0 (medianoche) a 23.')
            clase = st.selectbox('Clase de accidente', opciones['clases'],
                                 index=indice(opciones['clases'], 'Choque'),
                                 help='Cómo ocurrió el accidente: choque, atropello, caída de un ocupante, volcamiento u otro.')
            choque = st.selectbox('Choque con', opciones['choque'],
                                  index=indice(opciones['choque'], 'Vehiculo'),
                                  help='Con qué chocó el vehículo: otro vehículo, un objeto fijo o un animal. "Sin dato" si no fue registrado.')
            objeto = st.selectbox('Objeto fijo', opciones['objeto'],
                                  index=indice(opciones['objeto'], 'Ninguno'),
                                  help='Si el vehículo chocó contra algo fijo, como un poste, muro o árbol. "Ninguno" si no hubo.')
            via6 = st.selectbox('Tipo de vía (vía 6)', opciones['vias6'],
                                format_func=etiqueta_via6, index=indice(opciones['vias6'], 'Otro'),
                                help='Categoría de una de las vías del accidente.')
        with col2:
            hip1 = st.selectbox('Hipótesis 1 (código del informe)', opciones['hipotesis'],
                                format_func=etiqueta_hipotesis, index=indice(opciones['hipotesis'], 139),
                                help='Código de la primera posible causa, según el agente de tránsito que atendió el caso.')
            hip2 = st.selectbox('Hipótesis 2 (código del informe)', opciones['hipotesis'],
                                format_func=etiqueta_hipotesis, index=indice(opciones['hipotesis'], 0),
                                help='Código de una segunda posible causa. "Ninguna (0)" si el informe no tiene una segunda.')
            barrio = st.selectbox('Barrio', opciones['barrios'],
                                  index=indice(opciones['barrios'], 'CENTRO'),
                                  help='Barrio donde ocurrió el accidente.')
            localidad = st.selectbox('Localidad', opciones['localidades'],
                                     format_func=lambda v: f'Localidad {v}',
                                     index=indice(opciones['localidades'], 1),
                                     help='Localidad de Cartagena (1, 2 o 3) donde ocurrió el accidente.')
            lat = st.number_input('Latitud', min_value=LAT_RANGO[0], max_value=LAT_RANGO[1],
                                  value=CENTRO_CARTAGENA[0], step=0.001, format='%.5f',
                                  help='Coordenada norte-sur del punto del accidente. Entre 10,1 y 10,6 para Cartagena.')
            lon = st.number_input('Longitud', min_value=LON_RANGO[0], max_value=LON_RANGO[1],
                                  value=CENTRO_CARTAGENA[1], step=0.001, format='%.5f',
                                  help='Coordenada este-oeste del punto del accidente. Entre -75,8 y -75,2 para Cartagena. Por defecto es el centro de la ciudad.')
        enviado = st.form_submit_button('Calcular probabilidad', type='primary')

    if enviado:
        fila = construir_fila(fecha, hora, clase, choque, objeto, int(hip1), int(hip2),
                              barrio, int(localidad), via6, lat, lon)
        try:
            prob = float(modelo.predict_proba(fila)[0, 1])
        except Exception as error:
            st.error(f'No se pudo calcular la predicción: {error}')
            return
        res = cargar_resultados()
        umbral = res.get('umbral', 0.5)
        st.subheader('Resultado')
        c1, c2 = st.columns(2)
        c1.metric('Probabilidad de accidente fatal', f'{prob * 100:.1f} %')
        c2.metric('Predicción', 'Fatal' if prob >= umbral else 'No fatal')
        st.progress(prob)
        st.caption(f'El modelo predice "fatal" cuando la probabilidad es de {umbral:.0%} o más.')
        if fecha.year == 2025:
            tasa, minimo, maximo = cifras_2025(res)
            st.warning('Ojo: el año 2025 tiene una proporción de accidentes fatales muy distinta a los años anteriores '
                       f'({tasa} % frente a {minimo} %–{maximo} % entre 2020 y 2024), y el modelo no logra predecirlo '
                       'bien con datos de años previos. Toma esta probabilidad con mucha cautela.')


def predecir_lote(df, modelo):
    faltantes = [c for c in COLUMNAS_NECESARIAS if c not in df.columns]
    if faltantes:
        st.error('Al archivo le faltan estas columnas: ' + ', '.join(faltantes) +
                 '. Revisa que tenga los nombres exactos del CSV original.')
        return None
    if df.empty:
        st.error('El archivo no tiene filas.')
        return None

    lote = df.copy()
    for columna in ['latitud', 'longitud']:
        lote[columna] = pd.to_numeric(lote[columna], errors='coerce')
    for columna in ['hipotesis_1', 'hipotesis_2', 'localidad']:
        lote[columna] = pd.to_numeric(lote[columna], errors='coerce').fillna(0).astype(int)

    valida = (
        pd.to_datetime(lote['fecha'], errors='coerce').notna()
        & pd.to_datetime(lote['hora'], errors='coerce').notna()
        & lote['clase_accidente'].notna()
    ).to_numpy()
    if not valida.any():
        st.error('Ninguna fila tiene fecha, hora y clase de accidente válidas.')
        return None

    probabilidades = np.full(len(lote), np.nan)
    try:
        probabilidades[valida] = modelo.predict_proba(lote.loc[valida])[:, 1]
    except Exception as error:
        st.error(f'No se pudo calcular la predicción con este archivo: {error}')
        return None

    sin_prediccion = int((~valida).sum())
    if sin_prediccion:
        st.warning(f'{sin_prediccion} filas no tienen fecha, hora o clase de accidente válidas, '
                   'así que no tienen predicción.')

    res = cargar_resultados()
    filas_2025 = int((pd.to_datetime(lote['fecha'], errors='coerce').dt.year == 2025).sum())
    if filas_2025:
        tasa, minimo, maximo = cifras_2025(res)
        st.warning(f'Ojo: {filas_2025} filas tienen fecha de 2025, año que tiene una proporción de accidentes fatales '
                   f'({tasa} % frente a {minimo} %–{maximo} % entre 2020 y 2024) muy distinta a los años anteriores. '
                   'El modelo no logra predecirlo bien con datos de años previos, así que toma estas predicciones '
                   'con mucha cautela.')

    umbral = res.get('umbral', 0.5)
    resultado = df.copy()
    resultado['probabilidad_fatal'] = np.round(probabilidades, 4)
    resultado['prediccion'] = np.where(np.isnan(probabilidades), 'Sin predicción',
                                       np.where(probabilidades >= umbral, 'Fatal', 'No fatal'))
    return resultado


def pestana_lote(modelo):
    st.write('Sube un CSV con el mismo formato del archivo original (los nombres de columnas deben coincidir). '
             'Recibirás el mismo archivo con dos columnas nuevas: la probabilidad y la predicción.')
    st.caption('Columnas que el modelo necesita: ' + ', '.join(COLUMNAS_NECESARIAS))

    with st.expander('¿Qué debe traer el archivo y qué devuelve la app?'):
        st.markdown(
            '- **Formato:** el mismo CSV de accidentes de Cartagena, con los nombres de columna originales. '
            'Puedes usar el archivo completo o un extracto.\n'
            '- **Columnas obligatorias:** fecha, hora, clase de accidente, choque con, objeto fijo, hipótesis 1 y 2, '
            'barrio, localidad, vía 6, latitud y longitud.\n'
            '- **Columnas que la app agrega:** *probabilidad_fatal* es la probabilidad (entre 0 y 1) de que el accidente '
            'tenga al menos un muerto. *prediccion* dice "Fatal" si esa probabilidad es 0,5 o más, "No fatal" si es menor, '
            'y "Sin predicción" si la fila no tiene fecha, hora o clase válidas.\n'
            '- **Descarga:** al final puedes bajar el mismo archivo con esas dos columnas nuevas.'
        )

    archivo = st.file_uploader('Archivo CSV', type='csv')
    if archivo is None:
        return

    try:
        df = pd.read_csv(archivo)
    except Exception as error:
        st.error(f'No se pudo leer el archivo como CSV: {error}')
        return

    resultado = predecir_lote(df, modelo)
    if resultado is None:
        return

    st.success(f'Listo: {len(resultado)} filas procesadas.')
    columnas_vista = [c for c in ['ipat', 'fecha', 'hora', 'clase_accidente', 'probabilidad_fatal', 'prediccion']
                      if c in resultado.columns]
    st.write('Vista previa (primeras 10 filas):')
    st.dataframe(resultado[columnas_vista].head(10), hide_index=True, width='stretch')
    st.download_button('Descargar predicciones (CSV)',
                       data=resultado.to_csv(index=False).encode('utf-8-sig'),
                       file_name='predicciones.csv', mime='text/csv')


def pestana_acerca():
    res = cargar_resultados()
    mejor = res['seleccion']['mejor']
    test = res['modelos_finales'][mejor]['test']
    datos = res['datos']

    st.write(f'El mejor modelo fue **{mejor}**. Las métricas de abajo se calcularon sobre el 30% de los '
             'accidentes que el modelo no vio durante el entrenamiento.')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('F1 (clase fatal)', f"{test['f1']:.3f}")
    c2.metric('Precisión', f"{test['precision']:.3f}")
    c3.metric('Recall', f"{test['recall']:.3f}")
    c4.metric('ROC AUC', f"{test['roc_auc']:.3f}")

    st.markdown(
        '- **Precisión:** de los accidentes que el modelo marca como fatales, qué porcentaje lo fue.\n'
        '- **Recall:** de los accidentes fatales reales, qué porcentaje logra detectar el modelo.\n'
        '- **F1:** promedio entre precisión y recall. Sirve para comparar modelos cuando los datos están desbalanceados.\n'
        '- **ROC AUC:** qué tan bien separa el modelo los fatales de los no fatales (1 es perfecto, 0.5 es azar).'
    )

    st.subheader('Validación en 2025')
    st.write('La cifra de F1 de arriba es del test aleatorio y no refleja lo que pasa en 2025. Para probar eso, '
             'entrené con los accidentes de 2020 a 2024 y probé con los de 2025.')
    temporal = res['validacion_temporal']['modelos']
    nombres = {'Gradient Boosting': 'Boosting (Gradient Boosting)', 'Árbol de decisión': 'Árbol de decisión',
               'Regresión logística': 'Regresión logística'}
    filas_temporal = [
        {'Modelo': nombre, 'F1 (2025)': temporal[clave]['f1'], 'Recall (2025)': temporal[clave]['recall'],
         'ROC AUC (2025)': temporal[clave]['roc_auc']}
        for nombre, clave in nombres.items()
    ]
    st.dataframe(pd.DataFrame(filas_temporal).round(3), hide_index=True, width='stretch')
    st.write(res['interpretacion_temporal'])

    st.subheader('Datos')
    total = f"{datos['accidentes_consolidados']:,}".replace(',', '.')
    st.write(
        f"Se usaron {total} accidentes (una fila por accidente) de Cartagena, "
        f"entre 2020 y 2025. El {datos['porcentaje_fatal']:.1f}% fueron fatales. "
        'Se separaron en 70% para entrenar y 30% para probar.'
    )

    st.subheader('Comparación de modelos')
    filas = []
    for nombre, valores in res['modelos_finales'].items():
        filas.append({
            'Modelo': nombre,
            'F1 validación cruzada (70%)': valores['cv_f1_mean'],
            'F1 prueba (30%)': valores['test']['f1'],
            'Precisión': valores['test']['precision'],
            'Recall': valores['test']['recall'],
            'ROC AUC': valores['test']['roc_auc'],
        })
    tabla = pd.DataFrame(filas).sort_values('F1 validación cruzada (70%)', ascending=False)
    st.dataframe(tabla.round(3), hide_index=True, width='stretch')

    for figura in ['comparacion_modelos.png', 'curva_roc.png']:
        if (RUTA_FIGURAS / figura).exists():
            st.image(str(RUTA_FIGURAS / figura))

    st.info('Este modelo fue entrenado con datos públicos de un trabajo académico. Sus resultados no garantizan '
            'nada en la vida real y no deben usarse para decidir sobre tránsito, seguridad vial o emergencias.')


def main():
    st.title('Accidentes de tránsito con víctimas fatales en Cartagena')
    modelo = cargar_modelo()
    opciones = opciones_formulario()

    tab_predecir, tab_lote, tab_acerca = st.tabs(['Predecir un accidente', 'Predecir desde un CSV', 'Acerca del modelo'])
    with tab_predecir:
        pestana_predecir(modelo, opciones)
    with tab_lote:
        pestana_lote(modelo)
    with tab_acerca:
        pestana_acerca()


main()
