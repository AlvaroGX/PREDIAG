import json
import sqlite3
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules

SINTOMAS = [
    'enciende', 'sufrio_golpe', 'hace_ruido', 'sobrecalienta',
    'muy_lento', 'pantalla_negra', 'pantalla_azul', 'no_imprime',
    'atasca_papel', 'imprime_lineas', 'succion_bomba', 'no_enciende_del_todo',
    'reinicia_solo', 'error_sistema', 'pitidos_encendido',
    'funciona_sin_bateria', 'funciona_tras_golpe', 'muy_lento_post_formato'
]
FEATURES = ['tipo_equipo_cod'] + SINTOMAS

SEP = "=" * 60

print(SEP)
print("  ENTRENAMIENTO COMPLETO - SISTEMA PREDIAGNOSTICO")
print(SEP)

# ── 1. DATASET DESDE CONSULTA SQL A BD RELACIONAL ──────────────
print("\n=== 1. CARGA DE DATOS DESDE SQL ===\n")

conn = sqlite3.connect('historial.db')
df_csv = pd.read_csv('DATASET.csv')
df_csv.to_sql('dataset', conn, if_exists='replace', index=False)

CONSULTA_SQL = """
    SELECT id, tipo_equipo, enciende, sufrio_golpe, hace_ruido,
           sobrecalienta, muy_lento, pantalla_negra, pantalla_azul,
           no_imprime, atasca_papel, imprime_lineas, succion_bomba,
           no_enciende_del_todo, reinicia_solo, error_sistema,
           pitidos_encendido, funciona_sin_bateria, funciona_tras_golpe,
           muy_lento_post_formato, tipo_falla, reparacion, costo_aprox
    FROM dataset
"""
df = pd.read_sql_query(CONSULTA_SQL, conn)
conn.close()

print(f"  Consulta SQL ejecutada")
print(f"  Registros cargados: {len(df)}")
print(f"  Tipos de falla: {df['tipo_falla'].nunique()}")
print(f"  Equipos: {list(df['tipo_equipo'].unique())}\n")

# ── 2. USO EXPLICITO DE NUMPY ──────────────────────────────────
print("=== 2. NUMPY - PROCESAMIENTO NUMERICO ===\n")

matriz_sintomas = df[SINTOMAS].values
print(f"  Matriz numpy: forma={matriz_sintomas.shape}, dtype={matriz_sintomas.dtype}")
print(f"  Total de sintomas activos: {int(np.sum(matriz_sintomas))}")
print(f"  Promedio de sintomas por caso: {np.mean(np.sum(matriz_sintomas, axis=1)):.2f}")
print(f"  Matriz sin nulos: {not bool(np.any(np.isnan(matriz_sintomas.astype(float))))}")
print(f"  Costo promedio (numpy): S/{np.mean(df['costo_aprox'].values):.2f}")
print(f"  Costo std (numpy): S/{np.std(df['costo_aprox'].values):.2f}\n")

# ── 3. PREPARACION DE DATOS ────────────────────────────────────
print("=== 3. PREPARACION DE DATOS ===\n")

encoder_equipo = LabelEncoder()
df['tipo_equipo_cod'] = encoder_equipo.fit_transform(df['tipo_equipo'])
print(f"  Equipos codificados: {dict(zip(encoder_equipo.classes_, encoder_equipo.transform(encoder_equipo.classes_)))}")

X = df[FEATURES]
y = df['tipo_falla']
costos = df['costo_aprox'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
costos_train, costos_test = train_test_split(
    costos, test_size=0.2, random_state=42
)

print(f"  Train: {len(X_train)} | Test: {len(X_test)}\n")

# ── 4. CLASIFICACION: RANDOM FOREST ────────────────────────────
print("=== 4. RANDOM FOREST ===\n")

rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=4,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"  Accuracy: {acc_rf:.2%}")
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
cm_rf = confusion_matrix(y_test, y_pred_rf).tolist()

# ── 5. CLASIFICACION: KNN ─────────────────────────────────────
print("=== 5. K-NEAREST NEIGHBORS (KNN) ===\n")

knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"  K=5, distance-weighted, euclidean metric")
print(f"  Accuracy: {acc_knn:.2%}")

# ── 6. CLASIFICACION: ARBOL DE DECISION (ID3) ─────────────────
print("=== 6. ARBOL DE DECISION (ID3) ===\n")

dt = DecisionTreeClassifier(
    criterion='entropy', max_depth=10, min_samples_split=4,
    random_state=42
)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
print(f"  Criterio: entropy (ID3), max_depth=10")
print(f"  Accuracy: {acc_dt:.2%}")

importancias_dt = sorted(
    zip(FEATURES, dt.feature_importances_),
    key=lambda x: -x[1]
)
print("  Features mas importantes (ID3):")
for f, imp in importancias_dt[:5]:
    print(f"    {f}: {imp:.4f}")

# ── 7. RED NEURONAL CON OPTIMIZADOR ADAM ──────────────────────
print("\n=== 7. MLP CLASSIFIER + ADAM ===\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation='relu',
    solver='adam', max_iter=500, random_state=42,
    learning_rate_init=0.001, verbose=False
)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"  Arquitectura: 19->64->32->7 neuronas")
print(f"  Activacion: ReLU | Salida: Softmax")
print(f"  Optimizador: Adam (lr=0.001)")
print(f"  Accuracy: {acc_mlp:.2%}")

pesos_capa1 = mlp.coefs_[0]
sesgos_capa1 = mlp.intercepts_[0]
print(f"  Pesos capa 1: {pesos_capa1.shape}")
print(f"  Sesgos capa 1: {sesgos_capa1.shape}")
print(f"  Pesos capa 2: {mlp.coefs_[1].shape}")
print(f"  Sesgos capa 2: {mlp.intercepts_[1].shape}")

# ── 8. REGRESION LINEAL + MSE ─────────────────────────────────
print("\n=== 8. REGRESION LINEAL + MSE ===\n")

lr = LinearRegression()
lr.fit(X_train_scaled, costos_train)
costos_pred = lr.predict(X_test_scaled)
mse = mean_squared_error(costos_test, costos_pred)
rmse = np.sqrt(mse)
print(f"  Predice: costo_aprox (S/)")
print(f"  Coeficientes: {len(lr.coef_)} features")
print(f"  Intercepto: S/{lr.intercept_:.2f}")
print(f"  MSE (Mean Squared Error): {mse:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  Precio promedio real: S/{np.mean(costos_test):.2f}")

# ── 9. K-MEANS CLUSTERING ──────────────────────────────────────
print("\n=== 9. K-MEANS CLUSTERING ===\n")

kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

print(f"  Clusters formados: {kmeans.n_clusters}")
print("  Distribucion por cluster:")
for i in range(kmeans.n_clusters):
    idx = df['cluster'] == i
    tipos = df[idx]['tipo_falla'].value_counts()
    print(f"    Cluster {i}: {len(df[idx])} registros")
    for tipo, cnt in tipos.head(2).items():
        print(f"      -> {tipo}: {cnt}")

# ── 10. APRIORI ────────────────────────────────────────────────
print("\n=== 10. APRIORI - REGLAS DE ASOCIACION ===\n")

df_apriori = df[SINTOMAS + ['tipo_falla']].copy()
for s in SINTOMAS:
    df_apriori[s] = df_apriori[s].astype(bool)

df_apriori = pd.get_dummies(df_apriori, columns=['tipo_falla'], prefix='', prefix_sep='')
cols_bool = [c for c in df_apriori.columns if c.startswith(('Falla ', 'SO ', 'Sobre')) or c in SINTOMAS]

df_apriori_bool = df_apriori[cols_bool].astype(bool)

freq = apriori(df_apriori_bool, min_support=0.05, use_colnames=True)
print(f"  Itemsets frecuentes (min_support=0.05): {len(freq)}")

if len(freq) > 1:
    reglas = association_rules(freq, metric='confidence', min_threshold=0.5)
    reglas = reglas.sort_values('lift', ascending=False)
    print(f"  Reglas generadas: {len(reglas)}")
    print("\n  Top 5 reglas por lift:")
    for _, r in reglas.head(5).iterrows():
        ant = ', '.join(sorted(list(r['antecedents'])))[:50]
        con = ', '.join(sorted(list(r['consequents'])))[:50]
        print(f"    {{{ant}}} -> {{{con}}}")
        print(f"      Soporte={r['support']:.3f} Confianza={r['confidence']:.3f} Lift={r['lift']:.3f}")

# ── 11. COMPARATIVA Y GUARDADO ─────────────────────────────────
print("\n=== 11. RESULTADOS Y GUARDADO ===\n")

print("  COMPARATIVA DE ALGORITMOS:")
print(f"    {'Algoritmo':<25} {'Accuracy':<10}")
print(f"    {'-'*35}")
print(f"    {'Random Forest':<25} {acc_rf:.2%}")
print(f"    {'KNN (k=5)':<25} {acc_knn:.2%}")
print(f"    {'Arbol Decision (ID3)':<25} {acc_dt:.2%}")
print(f"    {'MLP + Adam':<25} {acc_mlp:.2%}")

mejor = max([(acc_rf, 'Random Forest'), (acc_knn, 'KNN'), (acc_dt, 'Arbol Decision'), (acc_mlp, 'MLP')])
print(f"\n  Mejor algoritmo: {mejor[1]} ({mejor[0]:.2%})")
print(f"  Guardando Random Forest como modelo principal...")

joblib.dump(rf, 'modelo.pkl')
joblib.dump(FEATURES, 'features.pkl')
joblib.dump(encoder_equipo, 'encoder_equipo.pkl')
joblib.dump(knn, 'modelo_knn.pkl')
joblib.dump(dt, 'modelo_dt.pkl')
joblib.dump(mlp, 'modelo_mlp.pkl')
joblib.dump(lr, 'modelo_lr.pkl')
joblib.dump(kmeans, 'modelo_kmeans.pkl')
joblib.dump(scaler, 'scaler.pkl')

metrics = {
    'accuracy': round(float(acc_rf), 4),
    'n_registros': len(df),
    'n_features': len(FEATURES),
    'n_clases': len(rf.classes_),
    'clases': list(rf.classes_),
    'classification_report': {
        k: v if isinstance(v, (int, float)) else {
            mk: round(float(mv), 4) for mk, mv in v.items()
        } for k, v in report_rf.items()
    },
    'confusion_matrix': cm_rf,
    'feature_importances': {
        f: round(float(i), 4) for f, i in
        sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])
    },
    'distribucion_fallas': df['tipo_falla'].value_counts().to_dict(),
    'distribucion_equipos': df['tipo_equipo'].value_counts().to_dict(),
    'comparativa_algoritmos': {
        'Random Forest': round(float(acc_rf), 4),
        'KNN': round(float(acc_knn), 4),
        'Arbol Decision ID3': round(float(acc_dt), 4),
        'MLP + Adam': round(float(acc_mlp), 4),
    },
    'mse': round(float(mse), 2),
    'rmse': round(float(rmse), 2),
    'regresion_lineal': {
        'intercepto': round(float(lr.intercept_), 2),
        'coeficientes': [round(float(c), 4) for c in lr.coef_],
    },
    'red_neuronal': {
        'arquitectura': '19->64->32->7',
        'activacion': 'ReLU',
        'optimizador': 'Adam',
        'pesos_capa1_shape': list(pesos_capa1.shape),
        'sesgos_capa1_shape': list(sesgos_capa1.shape),
    },
    'numpy_stats': {
        'matriz_shape': list(matriz_sintomas.shape),
        'sintomas_totales': int(np.sum(matriz_sintomas)),
        'sintomas_promedio': round(float(np.mean(np.sum(matriz_sintomas, axis=1))), 2),
        'costo_promedio': round(float(np.mean(df['costo_aprox'].values)), 2),
        'costo_std': round(float(np.std(df['costo_aprox'].values)), 2),
    },
    'apriori': {
        'itemsets_encontrados': len(freq) if len(freq) > 0 else 0,
        'reglas_generadas': len(reglas) if len(freq) > 1 else 0,
    },
}
with open('metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\n" + SEP)
print("  ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print(SEP)
