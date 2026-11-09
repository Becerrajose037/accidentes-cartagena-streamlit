# %% [markdown]
# # Predicción de accidentes de tránsito con víctimas mortales (Cartagena)
# Metodología CRISP-DM. Objetivo: predecir si un accidente deja al menos un muerto (`fatal`).
# Se evalúa con F1 de la clase positiva porque la exactitud sería engañosa (~82% de accidentes no son fatales).

# %%
import json
import urllib.request
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.contingency_tables import mcnemar

from prepro import (CATEGORICAS, LAT_RANGO, LON_RANGO, NUMERICAS, VARIABLES, consolidar, preparar)

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()
RUTA_CSV = BASE / 'accidentes_cartagena.csv'
URL_CSV = 'https://www.datos.gov.co/resource/5z62-bn99.csv?$limit=200000'
RUTA_FIGURAS = BASE / 'figuras'
RUTA_MODELO = BASE / 'modelo_accidentes.joblib'
RUTA_RESULTADOS = BASE / 'resultados.json'
RUTA_REPORTE = BASE / 'reporte_ydata_profiling.html'
RUTA_FIGURAS.mkdir(exist_ok=True)
SEMILLA = 42

# %% [markdown]
# ## 1. Comprensión y limpieza de datos
# Si `accidentes_cartagena.csv` no está en el directorio de trabajo (por ejemplo, en Colab), se descarga desde datos.gov.co.

# %%
if not RUTA_CSV.exists():
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
    raise SystemExit(f'Al CSV le faltan columnas del esquema esperado: {sorted(faltantes)}')
resultados = {'semilla': SEMILLA, 'fecha_ejecucion': datetime.now().isoformat(timespec='seconds'),
              'umbral': 0.5}

nulos_pct = (raw.isna().mean() * 100).round(2)
resultados['nulos_pct_antes_de_imputar'] = nulos_pct[nulos_pct > 0].to_dict()

# Hallazgo: `gravedad` no sirve como objetivo. Hay filas "Con Heridos" con muertos > 0,
# y variantes de escritura ("Con muertos", "CON HERIDOS").
gravedad_norm = raw['gravedad'].str.strip().str.lower()
gravedad_inconsistente = int(((gravedad_norm == 'con heridos') & (raw['muertos'] > 0)).sum())

fuera_lat = ~raw['latitud'].between(*LAT_RANGO)
fuera_lon = ~raw['longitud'].between(*LON_RANGO)

resultados['limpieza'] = {
    'duplicados_exactos_eliminados': int(raw.duplicated().sum()),
    'latitud_fuera_de_rango_a_nan': int(fuera_lat.sum()),
    'longitud_fuera_de_rango_a_nan': int(fuera_lon.sum()),
    'gravedad_con_heridos_pero_muertos_mayor_0': gravedad_inconsistente,
    'gravedad_valores': raw['gravedad'].value_counts().to_dict(),
    'ipat_con_dos_filas_crudas': int((raw['ipat'].value_counts() == 2).sum()),
}

datos = consolidar(raw)
datos['fatal'] = (datos['muertos'] > 0).astype(int)
resultados['datos'] = {
    'filas_csv': len(raw),
    'accidentes_consolidados': len(datos),
    'fatal_1': int(datos['fatal'].sum()),
    'fatal_0': int((datos['fatal'] == 0).sum()),
    'porcentaje_fatal': round(100 * datos['fatal'].mean(), 2),
}

# %%
conteo = datos['fatal'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(['No fatal (0)', 'Fatal (1)'], conteo.values, color=['#4C72B0', '#DD8452'])
for i, v in enumerate(conteo.values):
    ax.text(i, v, f'{v}\n({v / len(datos) * 100:.1f}%)', ha='center', va='bottom')
ax.set_title('Variable objetivo: accidentes consolidados')
ax.set_ylabel('Número de accidentes')
ax.set_ylim(0, conteo.max() * 1.15)
fig.tight_layout()
fig.savefig(RUTA_FIGURAS / 'distribucion_objetivo.png', dpi=150)
plt.close(fig)

# %% [markdown]
# ## 2. Partición y validación
# Split 70/30 estratificado. Toda la validación cruzada y la búsqueda de hiperparámetros se hace solo sobre el 70%.
# `muertos`, `heridos` y `gravedad` no entran como features (fuga de información).

# %%
X = datos.drop(columns=['fatal', 'muertos', 'heridos'])
y = datos['fatal']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEMILLA)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

resultados['split'] = {
    'train': len(X_train), 'test': len(X_test),
    'train_pct': 70, 'test_pct': 30,
    'train_fatal_1': int(y_train.sum()), 'test_fatal_1': int(y_test.sum()),
}
resultados['variables'] = {
    'numericas': NUMERICAS,
    'categoricas': CATEGORICAS,
    'excluidas_fuga': ['ipat', 'muertos', 'heridos', 'gravedad', 'coordenadas'],
    'excluidas_redundantes_o_poco_informativas': ['lugar', 'via1', 'via2', 'via3', 'via4', 'via5',
                                                  'hipotesis_3', 'hipotesis_4'],
}


# %%
def construir_pipeline(clasificador, variables_numericas=NUMERICAS):
    numericas = Pipeline([
        ('imputar', SimpleImputer(strategy='median')),
        ('escalar', StandardScaler()),
    ])
    categoricas = Pipeline([
        ('imputar', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('codificar', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=30,
                                    sparse_output=False)),
    ])
    preprocesamiento = ColumnTransformer([
        ('num', numericas, variables_numericas),
        ('cat', categoricas, CATEGORICAS),
    ])
    return ImbPipeline([
        ('preparar', FunctionTransformer(preparar)),
        ('preprocesar', preprocesamiento),
        ('balancear', SMOTE(random_state=SEMILLA)),
        ('clf', clasificador),
    ])


def cv_f1(modelo):
    puntajes = cross_val_score(modelo, X_train, y_train, cv=skf, scoring='f1', n_jobs=-1)
    return {'cv_f1_mean': float(puntajes.mean()), 'cv_f1_std': float(puntajes.std())}


# %% [markdown]
# ## 3. Modelos clásicos con hiperparámetros iniciales (CV sobre el 70%)

# %%
clasicos_iniciales = {
    'Regresión logística': LogisticRegression(C=1.0, max_iter=1000),
    'Árbol de decisión': DecisionTreeClassifier(random_state=SEMILLA),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=SEMILLA),
}
resultados['cv_inicial'] = {}
for nombre, clf in clasicos_iniciales.items():
    resultados['cv_inicial'][nombre] = cv_f1(construir_pipeline(clf))
    print(nombre, resultados['cv_inicial'][nombre])

# %% [markdown]
# ## 3b. Relaciones lineales y no lineales (solo sobre el 70%)
# Pearson mide relación lineal, Spearman relación monótona (basada en rangos) e información mutua cualquier dependencia.
# Si Spearman supera a Pearson, hay una relación monótona no lineal.

# %%
F_train = preparar(X_train)
correlacion_pearson = {v: float(F_train[v].corr(y_train, method='pearson')) for v in NUMERICAS}
correlacion_spearman = {v: float(F_train[v].corr(y_train, method='spearman')) for v in NUMERICAS}
resultados['correlacion_con_fatal'] = {
    'pearson': correlacion_pearson,
    'spearman': correlacion_spearman,
}
print('Pearson:', correlacion_pearson)
print('Spearman:', correlacion_spearman)

# %%
x = np.arange(len(NUMERICAS))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - 0.2, np.abs(list(correlacion_pearson.values())), 0.4, label='|Pearson| (lineal)')
ax.bar(x + 0.2, np.abs(list(correlacion_spearman.values())), 0.4, label='|Spearman| (monótona)')
ax.set_xticks(x)
ax.set_xticklabels(NUMERICAS, rotation=30, ha='right')
ax.set_ylabel('|correlación| con fatal')
ax.set_title('Variables numéricas: relación lineal vs monótona con fatal (70%)')
ax.legend()
fig.tight_layout()
fig.savefig(RUTA_FIGURAS / 'correlacion_lineal_vs_no_lineal.png', dpi=150)
plt.close(fig)

# %%
mediana = F_train[NUMERICAS].median()
codificado = pd.DataFrame(index=F_train.index)
codificado[NUMERICAS] = F_train[NUMERICAS].fillna(mediana)
codificado[CATEGORICAS] = OrdinalEncoder().fit_transform(F_train[CATEGORICAS].astype(str))
discretas = np.array([v not in ('latitud', 'longitud') for v in VARIABLES])
mi = mutual_info_classif(codificado[VARIABLES], y_train, discrete_features=discretas,
                         random_state=SEMILLA)
ranking_mi = pd.Series(mi, index=VARIABLES).sort_values(ascending=False)
resultados['informacion_mutua'] = {
    'nota': 'Calculada sobre el 70%. Categóricas codificadas con OrdinalEncoder y numéricas imputadas con la mediana. '
            'Las variables de alta cardinalidad (barrio, localidad) tienden a tener información mutua sobreestimada.',
    'ranking': {k: float(v) for k, v in ranking_mi.items()},
}
print(resultados['informacion_mutua']['ranking'])

# %%
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(ranking_mi.index[::-1], ranking_mi.values[::-1], color='#55A868')
ax.set_xlabel('Información mutua con fatal (nats)')
ax.set_title('Información mutua de cada variable con fatal (70%)')
fig.tight_layout()
fig.savefig(RUTA_FIGURAS / 'informacion_mutua.png', dpi=150)
plt.close(fig)

# %%
lr_cv = resultados['cv_inicial']['Regresión logística']['cv_f1_mean']
arbol_cv = resultados['cv_inicial']['Árbol de decisión']['cv_f1_mean']
conclusion = ('el árbol supera a la logística, así que hay relaciones no lineales (o interacciones entre variables)'
              if arbol_cv > lr_cv else
              'la logística iguala o supera al árbol, así que no hay evidencia de relaciones no lineales relevantes')
resultados['interpretacion_relaciones'] = (
    f'Sin ajustar, el árbol de decisión obtiene F1 CV {arbol_cv:.3f} y la regresión logística {lr_cv:.3f}; {conclusion}.'
)
print(resultados['interpretacion_relaciones'])

# %% [markdown]
# ## 4. Ajuste de hiperparámetros (scoring = F1, StratifiedKFold 5, solo sobre el 70%)

# %%
resultados['tuning'] = {}


def ajustar(nombre, clf, espacio, buscador=GridSearchCV, **extra):
    busqueda = buscador(construir_pipeline(clf), espacio, cv=skf, scoring='f1',
                        n_jobs=-1, **extra).fit(X_train, y_train)
    i = busqueda.best_index_
    mejores = {k.replace('clf__', ''): v for k, v in busqueda.best_params_.items()}
    resultados['tuning'][nombre] = {
        'metodo': buscador.__name__,
        'mejores_hiperparametros': mejores,
        'cv_f1_mean': float(busqueda.cv_results_['mean_test_score'][i]),
        'cv_f1_std': float(busqueda.cv_results_['std_test_score'][i]),
        'combinaciones_evaluadas': int(len(busqueda.cv_results_['params'])),
    }
    print(nombre, resultados['tuning'][nombre])
    return clone(clf).set_params(**mejores)


ajustables = {
    'Regresión logística': (LogisticRegression(max_iter=1000), {'clf__C': [0.01, 0.1, 1, 10, 100]},
                            GridSearchCV, {}),
    'Árbol de decisión': (DecisionTreeClassifier(random_state=SEMILLA),
                          {'clf__max_depth': [3, 5, 8, 12, None],
                           'clf__min_samples_leaf': [1, 5, 10, 20, 50]},
                          GridSearchCV, {}),
    'KNN': (KNeighborsClassifier(),
            {'clf__n_neighbors': [3, 5, 11, 21, 31], 'clf__weights': ['uniform', 'distance']},
            GridSearchCV, {}),
    'Boosting (Gradient Boosting)': (GradientBoostingClassifier(random_state=SEMILLA),
                                     {'clf__learning_rate': [0.02, 0.05, 0.1, 0.2],
                                      'clf__n_estimators': [100, 200, 300],
                                      'clf__max_depth': [2, 3, 4],
                                      'clf__subsample': [0.7, 0.85, 1.0]},
                                     RandomizedSearchCV, {'n_iter': 12, 'random_state': SEMILLA}),
}
ajustados = {nombre: ajustar(nombre, clf, espacio, buscador, **extra)
             for nombre, (clf, espacio, buscador, extra) in ajustables.items()}
lr_ajustada = ajustados['Regresión logística']
arbol_ajustado = ajustados['Árbol de decisión']
knn_ajustado = ajustados['KNN']
gb_ajustado = ajustados['Boosting (Gradient Boosting)']

# %% [markdown]
# ## 4b. Validación anidada (solo sobre el 70%)
# Outer StratifiedKFold(5); dentro de cada pliegue externo se repite la misma búsqueda de hiperparámetros
# (inner StratifiedKFold(5)). El F1 externo estima el rendimiento sin el sesgo de elegir hiperparámetros con los mismos folds.

# %%
externo = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA + 1)
resultados['nota_cv_anidada'] = (
    'Outer StratifiedKFold(5, random_state=43) y inner StratifiedKFold(5, random_state=42) sobre el 70%, '
    'misma rejilla/RandomizedSearchCV que en el ajuste. Los folds externos son distintos de los de ajuste '
    '(con la misma semilla coinciden y el resultado sería trivialmente igual). '
    'f1_externo_* es el F1 de la clase fatal en cada pliegue externo; f1_cv_no_anidado es el F1 CV del ajuste normal. '
    'La diferencia indica el sesgo optimista de no anidar.'
)
resultados['cv_anidada'] = {}


def cv_anidada(nombre, clf, espacio, buscador=GridSearchCV, **extra):
    puntajes = []
    for idx_entrena, idx_valida in externo.split(X_train, y_train):
        busqueda = buscador(construir_pipeline(clf), espacio, cv=skf, scoring='f1',
                            n_jobs=-1, **extra).fit(X_train.iloc[idx_entrena], y_train.iloc[idx_entrena])
        pred = busqueda.predict(X_train.iloc[idx_valida])
        puntajes.append(f1_score(y_train.iloc[idx_valida], pred, zero_division=0))
    puntajes = np.array(puntajes)
    no_anidado = resultados['tuning'][nombre]['cv_f1_mean']
    resultados['cv_anidada'][nombre] = {
        'f1_externo_mean': float(puntajes.mean()),
        'f1_externo_std': float(puntajes.std()),
        'f1_externo_por_pliegue': [float(p) for p in puntajes],
        'f1_cv_no_anidado': no_anidado,
        'sesgo_no_anidado_menos_externo': float(no_anidado - puntajes.mean()),
    }
    print(nombre, resultados['cv_anidada'][nombre])


for nombre, (clf, espacio, buscador, extra) in ajustables.items():
    cv_anidada(nombre, clf, espacio, buscador, **extra)

# %% [markdown]
# ## 5. Ensambles
# Votación soft con LR, árbol y KNN ya ajustados; bagging de árboles; boosting con Gradient Boosting ajustado.

# %%
resultados['nota_ensambles'] = (
    'Votación y Bagging reutilizan los hiperparámetros ajustados con los mismos folds (StratifiedKFold 5 sobre el 70%) '
    'y no tienen validación anidada completa; su F1 CV puede estar algo sobreestimado.'
)
modelos_finales = {
    'Regresión logística': lr_ajustada,
    'Árbol de decisión': arbol_ajustado,
    'KNN': knn_ajustado,
    'Naive Bayes': GaussianNB(),
    'SVM': clasicos_iniciales['SVM'],
    'Votación (LR+Árbol+KNN)': VotingClassifier([
        ('lr', clone(lr_ajustada)),
        ('arbol', clone(arbol_ajustado)),
        ('knn', clone(knn_ajustado)),
    ], voting='soft', n_jobs=-1),
    'Bagging (árboles)': BaggingClassifier(estimator=clone(arbol_ajustado), n_estimators=100,
                                           random_state=SEMILLA, n_jobs=-1),
    'Boosting (Gradient Boosting)': clone(gb_ajustado),
}
CLASICOS = list(clasicos_iniciales)

# %% [markdown]
# ## 6. Entrenamiento final y evaluación en el 30% de prueba

# %%
pipelines = {}
resultados['modelos_finales'] = {}
for nombre, clf in modelos_finales.items():
    pipe = construir_pipeline(clf)
    cv = cv_f1(pipe)
    pipe.fit(X_train, y_train)
    pipelines[nombre] = pipe
    resultados['modelos_finales'][nombre] = {
        'tipo': 'clasico' if nombre in CLASICOS else 'ensamble',
        **cv,
    }
    print(nombre, cv)


def puntaje(pipe, datos_x):
    if hasattr(pipe, 'predict_proba'):
        return pipe.predict_proba(datos_x)[:, 1]
    return pipe.decision_function(datos_x)


def evaluar(y_real, pred, score):
    return {
        'accuracy': float(accuracy_score(y_real, pred)),
        'precision': float(precision_score(y_real, pred, zero_division=0)),
        'recall': float(recall_score(y_real, pred, zero_division=0)),
        'f1': float(f1_score(y_real, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_real, score)),
        'pr_auc': float(average_precision_score(y_real, score)),
    }


predicciones = {}
puntajes = {}
for nombre, pipe in pipelines.items():
    predicciones[nombre] = pipe.predict(X_test)
    puntajes[nombre] = puntaje(pipe, X_test)
    resultados['modelos_finales'][nombre]['test'] = evaluar(y_test, predicciones[nombre], puntajes[nombre])
    print(nombre, resultados['modelos_finales'][nombre]['test'])

# %% [markdown]
# ## 7. Comparación de ensambles contra el mejor clásico

# %%
mejor_clasico = max(CLASICOS, key=lambda n: resultados['modelos_finales'][n]['cv_f1_mean'])
ref_cv = resultados['modelos_finales'][mejor_clasico]['cv_f1_mean']
ref_test = resultados['modelos_finales'][mejor_clasico]['test']['f1']
resultados['mejor_clasico'] = {'nombre': mejor_clasico, 'cv_f1_mean': ref_cv, 'test_f1': ref_test}

resultados['ensambles_vs_mejor_clasico'] = {}
for nombre, datos_modelo in resultados['modelos_finales'].items():
    if datos_modelo['tipo'] == 'ensamble':
        resultados['ensambles_vs_mejor_clasico'][nombre] = {
            'mejora_cv_f1': datos_modelo['cv_f1_mean'] - ref_cv,
            'mejora_test_f1': datos_modelo['test']['f1'] - ref_test,
        }
        print(nombre, resultados['ensambles_vs_mejor_clasico'][nombre])

# %% [markdown]
# ## 8. Selección del mejor modelo y prueba de McNemar
# El ranking se hace por F1 de validación cruzada (solo con el 70%), para no elegir el modelo mirando el test.
# McNemar compara los aciertos y errores de los dos primeros del ranking sobre el 30% de prueba.

# %%
ranking = sorted(resultados['modelos_finales'],
                 key=lambda n: resultados['modelos_finales'][n]['cv_f1_mean'], reverse=True)
mejor, segundo = ranking[0], ranking[1]


def prueba_mcnemar(nombre_a, nombre_b):
    acierta_a = predicciones[nombre_a] == y_test.values
    acierta_b = predicciones[nombre_b] == y_test.values
    tabla = np.array([
        [np.sum(acierta_a & acierta_b), np.sum(acierta_a & ~acierta_b)],
        [np.sum(~acierta_a & acierta_b), np.sum(~acierta_a & ~acierta_b)],
    ])
    res = mcnemar(tabla, exact=False, correction=True)
    return {
        'tabla': tabla.tolist(),
        'estadistico': float(res.statistic),
        'p_valor': float(res.pvalue),
        'exacta': False,
        'correccion_continuidad': True,
    }


resultados['seleccion'] = {
    'criterio': 'Mayor F1 medio en CV 5-fold sobre el 70% de entrenamiento',
    'ranking_cv': ranking,
    'mejor': mejor,
    'segundo': segundo,
    'mejor_en_test_por_f1': max(resultados['modelos_finales'],
                                key=lambda n: resultados['modelos_finales'][n]['test']['f1']),
    'mcnemar_mejor_vs_segundo': prueba_mcnemar(mejor, segundo),
}
print('Mejor:', mejor, '| Segundo:', segundo)
print('McNemar:', resultados['seleccion']['mcnemar_mejor_vs_segundo'])

# %%
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, predicciones[mejor]),
                       display_labels=['No fatal', 'Fatal']).plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title(f'Matriz de confusión (test): {mejor}', fontsize=9)
fig.tight_layout()
fig.savefig(RUTA_FIGURAS / 'matriz_confusion_final.png', dpi=150)
plt.close(fig)

# %%
fig, ax = plt.subplots(figsize=(7, 6))
for nombre in resultados['modelos_finales']:
    fpr, tpr, _ = roc_curve(y_test, puntajes[nombre])
    auc = resultados['modelos_finales'][nombre]['test']['roc_auc']
    ax.plot(fpr, tpr, label=f'{nombre} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
ax.set_xlabel('Tasa de falsos positivos')
ax.set_ylabel('Tasa de verdaderos positivos')
ax.set_title('Curva ROC en el conjunto de prueba')
ax.legend(fontsize=7, loc='lower right')
fig.tight_layout()
fig.savefig(RUTA_FIGURAS / 'curva_roc.png', dpi=150)
plt.close(fig)

# %%
nombres = list(resultados['modelos_finales'])
cv_medias = [resultados['modelos_finales'][n]['cv_f1_mean'] for n in nombres]
cv_desv = [resultados['modelos_finales'][n]['cv_f1_std'] for n in nombres]
f1_test = [resultados['modelos_finales'][n]['test']['f1'] for n in nombres]
x = np.arange(len(nombres))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, cv_medias, 0.4, yerr=cv_desv, capsize=3, label='F1 CV (70% entrenamiento)')
ax.bar(x + 0.2, f1_test, 0.4, label='F1 prueba (30%)')
ax.set_xticks(x)
ax.set_xticklabels(nombres, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('F1 (clase fatal)')
ax.set_ylim(0, 1.05)
ax.set_title('Comparación de modelos')
ax.legend(loc='upper center', ncol=2)
fig.tight_layout()
fig.savefig(RUTA_FIGURAS / 'comparacion_modelos.png', dpi=150)
plt.close(fig)

# %% [markdown]
# ## 9. Predicción de ejemplo
# Cinco accidentes del conjunto de prueba: 3 fatales y 2 no fatales.

# %%
ejemplos = pd.concat([
    X_test[y_test == 1].sample(3, random_state=SEMILLA),
    X_test[y_test == 0].sample(2, random_state=SEMILLA),
])
ejemplos_reales = y_test.loc[ejemplos.index].values
ejemplos_pred = pipelines[mejor].predict(ejemplos)
ejemplos_score = puntaje(pipelines[mejor], ejemplos)
tipo_score = 'probabilidad' if hasattr(pipelines[mejor], 'predict_proba') else 'puntaje'
features_ej = preparar(ejemplos)
etiqueta = {0: 'no fatal', 1: 'fatal'}

resultados['ejemplos_prediccion'] = []
for k, (idx, fila) in enumerate(features_ej.iterrows()):
    real, pred, s = int(ejemplos_reales[k]), int(ejemplos_pred[k]), float(ejemplos_score[k])
    texto = (f"Accidente a las {fila['hora']}:00 h, {fila['clase_accidente']}, choque con "
             f"{fila['choque_con']}, localidad {fila['localidad']}, barrio {fila['barrio']}. "
             f"El modelo predice {etiqueta[pred]} ({tipo_score} {s:.2f}); real: {etiqueta[real]}. "
             f"{'Acierto' if pred == real else 'Error'}.")
    resultados['ejemplos_prediccion'].append({
        'registro_prueba': int(ejemplos.loc[idx, 'ipat']),
        'real': real,
        'prediccion': pred,
        'puntaje': round(s, 4),
        'interpretacion': texto,
    })
    print(texto)

# %% [markdown]
# ## 9b. Validación temporal y ablación sin `anio` (diagnóstico, no cambia el modelo final)
# Mismos hiperparámetros finales. Entrenamiento con accidentes de 2020–2024 y prueba con 2025.

# %%
anios = preparar(X)['anio']
mascara_entreno = anios <= 2024
mascara_prueba = anios == 2025
tasa = datos.assign(anio=anios).groupby('anio')['fatal'].agg(['size', 'sum', 'mean'])
resultados['tasa_fatal_por_anio'] = {
    int(anio): {'accidentes': int(f['size']), 'fatales': int(f['sum']), 'tasa_pct': round(100 * f['mean'], 2)}
    for anio, f in tasa.iterrows()
}
print(resultados['tasa_fatal_por_anio'])

# %%
tasas_graf = tasa.loc[2020:2025]
fig, ax = plt.subplots(figsize=(7, 4.5))
barras = ax.bar(tasas_graf.index.astype(str), tasas_graf['mean'] * 100, color='#4C72B0')
for barra, valor in zip(barras, tasas_graf['mean'] * 100):
    ax.text(barra.get_x() + barra.get_width() / 2, valor, f'{valor:.1f}%', ha='center', va='bottom')
ax.set_ylabel('Accidentes fatales (%)')
ax.set_xlabel('Año')
ax.set_title('Tasa de accidentes con víctimas mortales por año')
ax.set_ylim(0, tasas_graf['mean'].max() * 100 * 1.15)
fig.tight_layout()
fig.savefig(RUTA_FIGURAS / 'tasa_fatal_por_anio.png', dpi=150)
plt.close(fig)


# %%
def ajustar_y_evaluar(clf, X_ent, y_ent, X_pru, y_pru, variables_numericas=NUMERICAS):
    pipe = construir_pipeline(clone(clf), variables_numericas).fit(X_ent, y_ent)
    return evaluar(y_pru, pipe.predict(X_pru), puntaje(pipe, X_pru))


finales_temporal = {
    'Regresión logística': ajustados['Regresión logística'],
    'Árbol de decisión': ajustados['Árbol de decisión'],
    'Boosting (Gradient Boosting)': ajustados['Boosting (Gradient Boosting)'],
}
resultados['validacion_temporal'] = {
    'corte': 'entrenamiento: anio <= 2024; prueba: anio == 2025',
    'n_train': int(mascara_entreno.sum()),
    'n_test': int(mascara_prueba.sum()),
    'fatal_train': int(y[mascara_entreno].sum()),
    'fatal_test': int(y[mascara_prueba].sum()),
    'tasa_fatal_train_pct': round(100 * y[mascara_entreno].mean(), 2),
    'tasa_fatal_test_pct': round(100 * y[mascara_prueba].mean(), 2),
    'modelos': {},
}
for nombre, clf in finales_temporal.items():
    resultados['validacion_temporal']['modelos'][nombre] = ajustar_y_evaluar(
        clf, X[mascara_entreno], y[mascara_entreno], X[mascara_prueba], y[mascara_prueba])
    print(nombre, resultados['validacion_temporal']['modelos'][nombre])

# %%
GB = 'Boosting (Gradient Boosting)'
sin_anio = [v for v in NUMERICAS if v != 'anio']
gb_ajustado_base = ajustados[GB]
sin_anio_aleatorio = ajustar_y_evaluar(gb_ajustado_base, X_train, y_train, X_test, y_test, sin_anio)
sin_anio_temporal = ajustar_y_evaluar(gb_ajustado_base, X[mascara_entreno], y[mascara_entreno],
                                      X[mascara_prueba], y[mascara_prueba], sin_anio)
con_anio_aleatorio = resultados['modelos_finales'][GB]['test']
con_anio_temporal = resultados['validacion_temporal']['modelos'][GB]
resultados['ablacion_sin_anio'] = {
    'modelo': GB,
    'nota': 'Diagnóstico: mismo Gradient Boosting final, quitando solo `anio` de las numéricas. No se guarda ni reemplaza el modelo final.',
    'aleatorio_70_30': {'con_anio': con_anio_aleatorio, 'sin_anio': sin_anio_aleatorio},
    'temporal_2025': {'con_anio': con_anio_temporal, 'sin_anio': sin_anio_temporal},
}
print(resultados['ablacion_sin_anio'])

# %%
f1_al, f1_25 = con_anio_aleatorio['f1'], con_anio_temporal['f1']
rec_al, rec_25 = con_anio_aleatorio['recall'], con_anio_temporal['recall']
auc_con, auc_sin = con_anio_temporal['roc_auc'], sin_anio_temporal['roc_auc']
tasa_train, tasa_test = resultados['validacion_temporal']['tasa_fatal_train_pct'], resultados['validacion_temporal']['tasa_fatal_test_pct']
if max(auc_con, auc_sin) < 0.70:
    veredicto = 'ninguna de las dos versiones discrimina bien en 2025 (ROC AUC < 0,70)'
elif auc_sin >= auc_con - 0.02:
    veredicto = 'sin `anio` se conserva la capacidad de discriminar (diferencia ≤ 0,02)'
else:
    veredicto = 'sin `anio` se pierde capacidad de discriminar (diferencia > 0,02)'
resultados['interpretacion_temporal'] = (
    f"Con el modelo final, el F1 baja de {f1_al:.3f} (test aleatorio 30%) a {f1_25:.3f} en 2025 y el recall de "
    f"{rec_al:.3f} a {rec_25:.3f}, mientras la tasa de fatales sube de {tasa_train:.1f}% (train, ≤2024) a {tasa_test:.1f}% "
    f"(2025); sin `anio` el ROC AUC en 2025 es {auc_sin:.3f} frente a {auc_con:.3f} con `anio`, y el PR AUC de referencia "
    f"(prevalencia) es {tasa_test / 100:.3f}, así que {veredicto}."
)
print(resultados['interpretacion_temporal'])

# %% [markdown]
# ## 10. Guardar modelo y resultados
# El Pipeline guardado incluye `preparar`, así que recibe el DataFrame crudo del CSV.

# %%
joblib.dump(pipelines[mejor], RUTA_MODELO)
resultados['archivos'] = {
    'modelo': RUTA_MODELO.name,
    'reporte_ydata': RUTA_REPORTE.name,
    'figuras': sorted(p.name for p in RUTA_FIGURAS.glob('*.png')),
}


def convertir(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


with open(RUTA_RESULTADOS, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=convertir)

print('Listo. Modelo guardado:', mejor)

# %% [markdown]
# ## 11. Reporte de perfilado (ydata-profiling)
# Perfilado del CSV crudo (antes de consolidar), para documentar la calidad de los datos.

# %%
from ydata_profiling import ProfileReport

reporte = ProfileReport(raw, title='Accidentes de transito Cartagena - Perfilado', minimal=False)
reporte.to_file(RUTA_REPORTE)
print('Reporte guardado:', RUTA_REPORTE)
