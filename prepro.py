"""Limpieza y feature engineering de los accidentes de tránsito de Cartagena.

`consolidar` pasa del CSV crudo (varias filas por accidente) a una fila por `ipat`.
`preparar` es fila a fila y no elimina filas: sirve tanto para el entrenamiento
como para la app, por eso va como primer paso del Pipeline guardado en joblib.
"""
import pandas as pd

LAT_RANGO = (10.1, 10.6)
LON_RANGO = (-75.8, -75.2)

NUMERICAS = ['hora', 'mes', 'anio', 'dia_semana', 'es_fin_de_semana', 'latitud', 'longitud']
CATEGORICAS = ['clase_accidente', 'choque_con', 'objeto_fijo', 'hipotesis_1', 'hipotesis_2',
               'barrio', 'localidad', 'via6', 'franja_horaria']
VARIABLES = NUMERICAS + CATEGORICAS
COLUMNAS_ENTRADA = ['fecha', 'hora', 'clase_accidente', 'choque_con', 'objeto_fijo', 'hipotesis_1',
                    'hipotesis_2', 'barrio', 'localidad', 'via6', 'latitud', 'longitud']


def _texto(serie):
    serie = serie.astype('string').str.strip()
    serie = serie.str.replace('\u00c3\u00b1', '\u00f1', regex=False)
    return serie.str.lower().str.capitalize()


def _horas(serie):
    """Acepta horas como '1899-12-31T17:30:00.000', '17:30' o el entero 17."""
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors='coerce').where(lambda s: s.between(0, 23))
    return pd.to_datetime(serie, errors='coerce', format='mixed').dt.hour


def consolidar(df):
    """Quita duplicados exactos y deja una fila por ipat. Muertos y heridos se suman."""
    d = df.drop_duplicates()
    sumas = d.groupby('ipat', as_index=False)[['muertos', 'heridos']].sum()
    primeras = d.drop_duplicates('ipat', keep='first').drop(columns=['muertos', 'heridos'])
    return primeras.merge(sumas, on='ipat', how='left')


def preparar(df):
    faltantes = [c for c in COLUMNAS_ENTRADA if c not in df.columns]
    if faltantes:
        raise ValueError('Faltan columnas requeridas: ' + ', '.join(faltantes))

    d = pd.DataFrame(index=df.index)

    fecha = pd.to_datetime(df['fecha'], errors='coerce', format='mixed')
    hora = _horas(df['hora'])
    d['hora'] = hora
    d['franja_horaria'] = pd.cut(
        hora, bins=[-1, 5, 11, 17, 23], labels=['Madrugada', 'Mañana', 'Tarde', 'Noche']
    ).astype('string')
    d['mes'] = fecha.dt.month
    d['anio'] = fecha.dt.year
    d['dia_semana'] = fecha.dt.dayofweek
    d['es_fin_de_semana'] = (fecha.dt.dayofweek >= 5).astype(int)

    lat = pd.to_numeric(df['latitud'], errors='coerce')
    lon = pd.to_numeric(df['longitud'], errors='coerce')
    d['latitud'] = lat.where(lat.between(*LAT_RANGO))
    d['longitud'] = lon.where(lon.between(*LON_RANGO))

    d['clase_accidente'] = _texto(df['clase_accidente'])
    d['choque_con'] = _texto(df['choque_con']).fillna('Desconocido')
    d['objeto_fijo'] = _texto(df['objeto_fijo']).replace({'0': 'Ninguno', 'No': 'Ninguno'}).fillna('Desconocido')
    d['hipotesis_1'] = df['hipotesis_1'].astype(str)
    d['hipotesis_2'] = df['hipotesis_2'].astype(str)
    d['barrio'] = df['barrio'].astype('string').str.strip().str.upper().fillna('Desconocido')
    d['localidad'] = df['localidad'].astype(str)
    d['via6'] = _texto(df['via6']).fillna('Desconocido')

    return d[VARIABLES]
