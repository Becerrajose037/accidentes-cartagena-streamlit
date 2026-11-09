"""Genera crisp_dm_accidentes.ipynb a partir de entrenar_modelo.py.

Las celdas se separan por marcadores `# %%`; `# %% [markdown]` produce una celda de texto.
El contenido de prepro.py se incluye como primera celda de código para que el notebook
funcione solo, por ejemplo en Google Colab (donde basta subir el CSV).

El notebook no escribe `modelo_accidentes.joblib`: lo guarda como `modelo_colab.joblib`
para no reemplazar el modelo portable que usa app.py (ver `NOTA_MODELO`).
"""
import json
import uuid
from pathlib import Path

BASE = Path(__file__).resolve().parent
ORIGEN = BASE / 'entrenar_modelo.py'
PREPRO = BASE / 'prepro.py'
DESTINO = BASE / 'crisp_dm_accidentes.ipynb'
LINEA_IMPORT_PREPRO = 'from prepro import'
MODELO_NOTEBOOK = 'modelo_colab.joblib'
INSTALACION = [
    '!pip install -q "numpy==2.3.5" "pandas==2.3.3" "scikit-learn==1.9.1" "imbalanced-learn==0.14.2" '
    '"joblib==1.6.0" "matplotlib==3.10.0" "scipy==1.16.3" "statsmodels==0.15.0" '
    '"ydata-profiling==4.18.4" "setuptools==80.9.0"',
]

DESCARGA_ORIGINAL = '''if not RUTA_CSV.exists():
    print('Descargando', URL_CSV)
    try:
        with urllib.request.urlopen(URL_CSV, timeout=300) as respuesta:
            contenido = respuesta.read()
        if len(contenido) < 10_000:
            raise ValueError(f'La descarga pesa solo {len(contenido)} bytes; no parece el CSV completo.')
        RUTA_CSV.write_bytes(contenido)
    except Exception as error:
        raise SystemExit(
            f'No se pudo descargar el dataset ({error}). '
            'Descárgalo a mano desde https://www.datos.gov.co/d/5z62-bn99 y guárdalo como '
            f'{RUTA_CSV.name} en esta carpeta.'
        ) from error

COLUMNAS_ESPERADAS = {'ipat', 'fecha', 'hora', 'gravedad', 'clase_accidente', 'choque_con',
                      'hipotesis_1', 'hipotesis_2', 'barrio', 'localidad', 'muertos', 'heridos',
                      'latitud', 'longitud'}
raw = pd.read_csv(RUTA_CSV)
faltantes = COLUMNAS_ESPERADAS - set(raw.columns)
if faltantes:
    raise SystemExit(f'Al CSV le faltan columnas del esquema esperado: {sorted(faltantes)}')'''

# La descarga ya está endurecida en entrenar_modelo.py, así que el notebook usa el mismo bloque.
DESCARGA_NOTEBOOK = DESCARGA_ORIGINAL

REEMPLAZOS = [
    ('RUTA_MODELO = BASE / \'modelo_accidentes.joblib\'',
     f'RUTA_MODELO = BASE / \'{MODELO_NOTEBOOK}\''),
]

NOTA_MODELO = [
    '## Nota: este cuaderno NO escribe `modelo_accidentes.joblib`',
    '',
    f'El Pipeline se guarda como `{MODELO_NOTEBOOK}`. El motivo es que el Pipeline contiene la función '
    '`preparar` definida en `__main__` (en Colab, el espacio de nombres del cuaderno), y `app.py` no puede '
    'deserializarla desde otro proceso. El archivo que carga la app, `modelo_accidentes.joblib`, se genera '
    'con `python entrenar_modelo.py`, que importa `preparar` desde `prepro.py` y por eso sí es portable. '
    f'`resultados.json` y las figuras de `figuras/` sí se guardan en el directorio actual y sirven para revisar '
    'el proceso, pero no reemplazan los del repositorio.',
]


def bloques(texto):
    actuales = []
    for linea in texto.splitlines():
        if linea.startswith('# %%'):
            actuales.append({'markdown': '[markdown]' in linea, 'lineas': []})
        elif actuales:
            actuales[-1]['lineas'].append(linea)
    return actuales


def celda_markdown(lineas):
    texto = [l[2:] if l.startswith('# ') else '' for l in lineas]
    return {'cell_type': 'markdown', 'metadata': {}, 'source': texto}


def celda_codigo(lineas):
    lineas = [l for l in lineas if not l.startswith(LINEA_IMPORT_PREPRO)]
    while lineas and not lineas[-1].strip():
        lineas.pop()
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': lineas}


def formato_source(celda):
    lineas = celda['source']
    celda['source'] = [l + '\n' for l in lineas[:-1]] + lineas[-1:]
    celda['id'] = uuid.uuid4().hex[:12]
    return celda


def reemplazar(celdas, viejo, nuevo):
    """Sustituye un bloque de la fuente en la primera celda de código que lo contenga."""
    for celda in celdas:
        if celda['cell_type'] != 'code':
            continue
        texto = '\n'.join(celda['source'])
        if viejo in texto:
            celda['source'] = texto.replace(viejo, nuevo).split('\n')
            return
    raise SystemExit('No se encontró en el notebook el bloque a reemplazar:\n' + viejo[:120])


def main():
    celdas = [
        celda_markdown([
            '# Cómo ejecutar este cuaderno (Google Colab)',
            '',
            '# Notebook generado desde `entrenar_modelo.py`. Si `accidentes_cartagena.csv` no está en el directorio '
            'de trabajo (en Colab: `/content`), se descarga automáticamente desde '
            '`https://www.datos.gov.co/resource/5z62-bn99.csv?$limit=200000`. Si la descarga falla, el cuaderno '
            'muestra un mensaje para subir `accidentes_cartagena.csv` manualmente.',
            '#',
            f'# Los resultados y las figuras se guardan en el directorio actual. El modelo se guarda como '
            f'`{MODELO_NOTEBOOK}` (ver la nota de la sección 10): el modelo que usa la app lo genera '
            '`entrenar_modelo.py`.',
        ]),
        celda_markdown([
            '# ## Instalación de dependencias',
            '',
            '# Versiones fijadas. Si Colab pide reiniciar el entorno tras instalar, reinícialo y ejecuta de nuevo '
            'desde la celda siguiente (no hace falta volver a instalar).',
        ]),
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
         'source': INSTALACION},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
         'source': PREPRO.read_text(encoding='utf-8').rstrip().splitlines()},
    ]
    for bloque in bloques(ORIGEN.read_text(encoding='utf-8')):
        if bloque['markdown']:
            celdas.append(celda_markdown(bloque['lineas']))
        elif any(l.strip() for l in bloque['lineas']):
            celdas.append(celda_codigo(bloque['lineas']))

    for viejo, nuevo in REEMPLAZOS:
        reemplazar(celdas, viejo, nuevo)

    indice = next(i for i, c in enumerate(celdas)
                  if c['cell_type'] == 'markdown' and any('## 10. Guardar modelo' in l for l in c['source']))
    celdas.insert(indice + 1, {'cell_type': 'markdown', 'metadata': {}, 'source': NOTA_MODELO})

    cuaderno = {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'colab': {'provenance': []},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'cells': [formato_source(c) for c in celdas],
    }
    DESTINO.write_text(json.dumps(cuaderno, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Notebook escrito:', DESTINO, 'celdas:', len(celdas))


if __name__ == '__main__':
    main()
