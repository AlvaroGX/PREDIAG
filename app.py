import json
import logging
import sqlite3
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, make_response
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DATOS = 'DATASET.csv'
MODELO_PATH = 'modelo.pkl'
FEATURES_PATH = 'features.pkl'
ENCODER_PATH = 'encoder_equipo.pkl'
DB_PATH = 'historial.db'

SINTOMAS_BINARIOS = [
    'enciende', 'sufrio_golpe', 'hace_ruido', 'sobrecalienta',
    'muy_lento', 'pantalla_negra', 'pantalla_azul', 'no_imprime',
    'atasca_papel', 'imprime_lineas', 'succion_bomba', 'no_enciende_del_todo',
    'reinicia_solo', 'error_sistema', 'pitidos_encendido',
    'funciona_sin_bateria', 'funciona_tras_golpe', 'muy_lento_post_formato'
]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS predicciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        equipo TEXT NOT NULL,
        sintomas TEXT NOT NULL,
        tipo_falla TEXT NOT NULL,
        reparacion TEXT NOT NULL,
        costo REAL,
        confianza REAL
    )''')
    conn.commit()
    conn.close()


def guardar_historial(equipo, sintomas, tipo_falla, reparacion, costo, confianza):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO predicciones (timestamp, equipo, sintomas, tipo_falla, reparacion, costo, confianza) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (datetime.now().isoformat(), equipo, json.dumps(sintomas), tipo_falla, reparacion, costo, confianza)
    )
    conn.commit()
    conn.close()


def obtener_historial(limite=50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT * FROM predicciones ORDER BY id DESC LIMIT ?', (limite,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


COSTOS_FIJO = {
    'Falla de hardware': 240.00,
    'Falla de pantalla': 200.00,
    'Falla de software': 40.00,
    'Sobrecalentamiento': 40.00,
    'Falla de impresión': 40.00,
    'Falla de energía': 40.00,
    'SO incompatible': 40.00,
    'Falla de teclado': 140.00,
}

NOTA_VENTA = {
    'Falla de hardware': 'Revisión general + diagnóstico de componentes',
    'Falla de pantalla': 'Cambio de pantalla + mano de obra',
    'Falla de software': 'Formateo e instalación de sistema operativo',
    'Sobrecalentamiento': 'Limpieza interna + cambio de pasta térmica',
    'Falla de impresión': 'Limpieza de cabezales y rodillos',
    'Falla de energía': 'Revisión de fuente de poder',
    'SO incompatible': 'Reinstalación con sistema operativo compatible',
    'Falla de teclado': 'Cambio de teclado + mano de obra',
}

REPARACIONES = {
    ('Falla de energía', 'Laptop'): 'Revisión de fuente o cargador',
    ('Falla de energía', 'PC de escritorio'): 'Revisión de fuente de poder',
    ('Falla de hardware', 'Laptop'): 'Revisión y cambio de componente',
    ('Falla de hardware', 'PC de escritorio'): 'Revisión y cambio de componente',
    ('Falla de pantalla', 'Laptop'): 'Cambio de pantalla',
    ('Falla de pantalla', 'PC de escritorio'): 'Cambio de pantalla',
    ('Falla de software', 'Laptop'): 'Formateo y reinstalación',
    ('Falla de software', 'PC de escritorio'): 'Formateo y reinstalación',
    ('Sobrecalentamiento', 'Laptop'): 'Limpieza interna y pasta térmica',
    ('Sobrecalentamiento', 'PC de escritorio'): 'Limpieza interna y pasta térmica',
    ('Falla de impresión', 'Impresora Epson'): 'Limpieza de cabezales o rodillos',
    ('SO incompatible', 'Laptop'): 'Reinstalación con SO compatible',
    ('SO incompatible', 'PC de escritorio'): 'Reinstalación con SO compatible',
    ('Falla de teclado', 'Laptop'): 'Cambio de teclado',
}


def cargar_recursos():
    modelo = joblib.load(MODELO_PATH)
    features = joblib.load(FEATURES_PATH)
    encoder = joblib.load(ENCODER_PATH)
    df = pd.read_csv(BASE_DATOS)
    ref = df.groupby('tipo_falla').agg(
        reparacion=('reparacion', lambda x: x.mode().iloc[0] if not x.mode().empty else '—'),
    ).to_dict('index')
    log.info('Recursos cargados: modelo, %d features, %d referencias', len(features), len(ref))
    return modelo, features, encoder, ref


try:
    modelo, features, encoder_equipo, ref_fallas = cargar_recursos()
    init_db()
except Exception as e:
    log.critical('Error al cargar recursos: %s', e)
    raise


def validar_sintomas(form):
    errores = []
    if form.get('equipo') not in encoder_equipo.classes_:
        errores.append('Equipo no válido')
    for s in SINTOMAS_BINARIOS:
        val = form.get(s)
        if val is None or val not in ('0', '1'):
            errores.append(f'Síntoma "{s}" inválido o faltante')
    return errores


def obtener_sintomas_dict(form):
    return {s: int(form[s]) for s in SINTOMAS_BINARIOS}


def predecir_falla(equipo, sintomas_dict):
    sintomas_valores = [sintomas_dict[s] for s in SINTOMAS_BINARIOS]
    equipo_codificado = encoder_equipo.transform([equipo])[0]
    datos = pd.DataFrame([[equipo_codificado, *sintomas_valores]], columns=features)
    tipo_falla = modelo.predict(datos)[0]
    probas = modelo.predict_proba(datos)[0]
    idx_clase = list(modelo.classes_).index(tipo_falla)
    confianza = round(float(probas[idx_clase] * 100), 1)
    reparacion = REPARACIONES.get((tipo_falla, equipo), '—')
    costo = COSTOS_FIJO.get(tipo_falla, 40.00)
    nota = NOTA_VENTA.get(tipo_falla, 'Servicio técnico general')
    return tipo_falla, reparacion, costo, confianza, nota


def obtener_importancias(sintomas_dict):
    fi = modelo.feature_importances_
    parejas = list(zip(features, fi))
    parejas = [p for p in parejas if p[0] != 'tipo_equipo_cod']
    parejas.sort(key=lambda x: -x[1])
    top = []
    for feat, imp in parejas[:5]:
        valor = sintomas_dict.get(feat, 0)
        top.append({'sintoma': feat, 'importancia': round(imp, 4), 'presente': bool(valor)})
    return top


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predecir', methods=['POST'])
def predecir():
    errores = validar_sintomas(request.form)
    if errores:
        log.warning('Validación fallida: %s', errores)
        return render_template('index.html', error='Datos inválidos. Verifique el formulario.')

    try:
        equipo = request.form['equipo']
        sintomas = obtener_sintomas_dict(request.form)
        teclado_falla = request.form.get('teclado_falla') == '1'

        hallazgos_extra = []
        if equipo == 'Laptop' and teclado_falla:
            hallazgos_extra.append({
                'tipo_falla': 'Falla de teclado',
                'reparacion': 'Cambio de teclado',
                'costo': 140.00,
                'nota': 'Cambio de teclado + mano de obra'
            })

        tipo_falla, reparacion, costo, confianza, nota = predecir_falla(equipo, sintomas)

        costo_total = costo + sum(h['costo'] for h in hallazgos_extra)
        costo_texto = f'S/{costo_total:.2f}' if costo_total else '—'
        importancias = obtener_importancias(sintomas)

        guardar_historial(equipo, sintomas, tipo_falla, reparacion, costo_total, confianza)
        log.info('Predicción: %s -> %s (%.1f%%)', equipo, tipo_falla, confianza)

        return render_template(
            'index.html',
            tipo_falla=tipo_falla,
            reparacion=reparacion,
            costo=costo_texto,
            confianza=confianza,
            importancias=importancias,
            nota=nota,
            hallazgos_extra=hallazgos_extra
        )

    except Exception as e:
        log.error('Error en predicción: %s', e)
        return render_template('index.html', error='Error al procesar la solicitud.')


@app.route('/api/predecir', methods=['POST'])
def api_predecir():
    data = request.get_json(silent=True) or request.form
    errores = validar_sintomas(data)
    if errores:
        return jsonify({'error': errores}), 400

    try:
        equipo = data['equipo']
        sintomas = obtener_sintomas_dict(data)
        teclado_falla = data.get('teclado_falla') == '1'

        hallazgos_extra = []
        if equipo == 'Laptop' and teclado_falla:
            hallazgos_extra.append({
                'tipo_falla': 'Falla de teclado',
                'reparacion': 'Cambio de teclado',
                'costo': 140.00,
                'nota': 'Cambio de teclado + mano de obra'
            })

        tipo_falla, reparacion, costo, confianza, nota = predecir_falla(equipo, sintomas)

        costo_total = costo + sum(h['costo'] for h in hallazgos_extra)
        importancias = obtener_importancias(sintomas)

        guardar_historial(equipo, sintomas, tipo_falla, reparacion, costo_total, confianza)
        log.info('API -> %s -> %s (%.1f%%)', equipo, tipo_falla, confianza)

        return jsonify({
            'equipo': equipo,
            'tipo_falla': tipo_falla,
            'reparacion': reparacion,
            'costo': costo_total,
            'confianza': confianza,
            'importancias': importancias,
            'nota': nota,
            'hallazgos_extra': hallazgos_extra,
        })

    except Exception as e:
        log.error('Error en API: %s', e)
        return jsonify({'error': 'Error interno'}), 500


@app.route('/historial')
def historial():
    registros = obtener_historial()
    return render_template('historial.html', registros=registros)


@app.route('/exportar', methods=['POST'])
def exportar():
    tipo_falla = request.form.get('tipo_falla', '—')
    reparacion = request.form.get('reparacion', '—')
    costo = request.form.get('costo', '—')
    confianza = request.form.get('confianza', '—')
    equipo = request.form.get('equipo', '—')
    nota = request.form.get('nota', '—')
    hallazgos_extra = request.form.get('hallazgos_extra', '')

    extra_texto = ''
    if hallazgos_extra:
        try:
            extra = json.loads(hallazgos_extra)
            for h in extra:
                extra_texto += f"\n+ {h['tipo_falla']}: {h['reparacion']} (S/{h['costo']:.2f})"
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    texto = f"""=== DIAGNÓSTICO DE EQUIPO ===
Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Equipo: {equipo}

Tipo de falla: {tipo_falla}
Reparación sugerida: {reparacion}
{extra_texto}
🧾 Servicios incluidos:
{nota}

Costo total: {costo}
Confianza: {confianza}%
=========================="""

    resp = make_response(texto)
    resp.headers['Content-Type'] = 'text/plain; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename=diagnostico.txt'
    return resp


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/dashboard')
def api_dashboard():
    try:
        with open('metrics.json', 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({'error': 'Ejecute train.py primero'}), 500

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total_pred = conn.execute('SELECT COUNT(*) as c FROM predicciones').fetchone()['c']
    avg_conf = conn.execute('SELECT ROUND(AVG(confianza),1) as c FROM predicciones').fetchone()['c']
    fallas_hist = conn.execute('SELECT tipo_falla, COUNT(*) as c FROM predicciones GROUP BY tipo_falla ORDER BY c DESC').fetchall()
    equipos_hist = conn.execute('SELECT equipo, COUNT(*) as c FROM predicciones GROUP BY equipo ORDER BY c DESC').fetchall()
    ultimas = conn.execute('SELECT timestamp, tipo_falla, confianza FROM predicciones ORDER BY id DESC LIMIT 10').fetchall()

    conn.close()

    return jsonify({
        'metrics': metrics,
        'total_predicciones': total_pred,
        'confianza_promedio': avg_conf,
        'fallas_historial': [dict(r) for r in fallas_hist],
        'equipos_historial': [dict(r) for r in equipos_hist],
        'ultimas_predicciones': [dict(r) for r in ultimas],
    })


@app.route('/modelos')
def modelos():
    try:
        with open('metrics.json', 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return render_template('modelos.html', error='Ejecute train.py primero')

    return render_template('modelos.html', metrics=metrics)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
