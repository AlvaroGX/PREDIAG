import pandas as pd
import numpy as np

df = pd.read_csv('DATASET.csv')

np.random.seed(123)

SINTOMAS = [
    'enciende', 'sufrio_golpe', 'hace_ruido', 'sobrecalienta',
    'muy_lento', 'pantalla_negra', 'pantalla_azul', 'no_imprime',
    'atasca_papel', 'imprime_lineas', 'succion_bomba', 'no_enciende_del_todo',
    'reinicia_solo', 'error_sistema', 'pitidos_encendido',
    'funciona_sin_bateria', 'funciona_tras_golpe', 'muy_lento_post_formato'
]

PERFIL_FALLAS = {
    'Falla de energía': {
        'equipos': ['Laptop', 'PC de escritorio'],
        'sintomas': {
            'enciende': 0.1, 'no_enciende_del_todo': 0.3, 'reinicia_solo': 0.2,
            'funciona_sin_bateria': 0.4, 'funciona_tras_golpe': 0.1,
        },
        'reparacion': 'Revisión de fuente o cargador',
        'costo': (40, 55)
    },
    'Falla de hardware': {
        'equipos': ['Laptop', 'PC de escritorio'],
        'sintomas': {
            'enciende': 0.1, 'sufrio_golpe': 0.6, 'hace_ruido': 0.6,
            'no_enciende_del_todo': 0.3, 'pitidos_encendido': 0.5,
            'funciona_tras_golpe': 0.3,
        },
        'reparacion': 'Revisión y cambio de componente',
        'costo': (200, 900)
    },
    'Falla de impresión': {
        'equipos': ['Impresora Epson'],
        'sintomas': {
            'enciende': 0.9, 'no_imprime': 0.8, 'atasca_papel': 0.5,
            'imprime_lineas': 0.5, 'succion_bomba': 0.4, 'hace_ruido': 0.5,
        },
        'reparacion': 'Limpieza de cabezales o rodillos',
        'costo': (35, 60)
    },
    'Falla de pantalla': {
        'equipos': ['Laptop', 'PC de escritorio'],
        'sintomas': {
            'enciende': 0.7, 'pantalla_negra': 0.8, 'pantalla_azul': 0.1,
            'sufrio_golpe': 0.3, 'funciona_tras_golpe': 0.2,
        },
        'reparacion': 'Cambio de pantalla',
        'costo': (180, 200)
    },
    'Falla de software': {
        'equipos': ['Laptop', 'PC de escritorio'],
        'sintomas': {
            'enciende': 0.9, 'muy_lento': 0.6, 'reinicia_solo': 0.5,
            'error_sistema': 0.5, 'pantalla_azul': 0.3,
            'funciona_sin_bateria': 0.2, 'funciona_tras_golpe': 0.1,
        },
        'reparacion': 'Formateo y reinstalación',
        'costo': (40, 60)
    },
    'SO incompatible': {
        'equipos': ['Laptop', 'PC de escritorio'],
        'sintomas': {
            'enciende': 0.9, 'muy_lento': 0.4, 'error_sistema': 0.3,
            'funciona_sin_bateria': 0.2, 'muy_lento_post_formato': 0.7,
            'funciona_tras_golpe': 0.1,
        },
        'reparacion': 'Reinstalación con SO compatible',
        'costo': (40, 60)
    },
    'Sobrecalentamiento': {
        'equipos': ['Laptop', 'PC de escritorio'],
        'sintomas': {
            'enciende': 0.8, 'sobrecalienta': 0.9, 'muy_lento': 0.6,
            'reinicia_solo': 0.4, 'hace_ruido': 0.3, 'error_sistema': 0.2,
        },
        'reparacion': 'Limpieza interna y pasta térmica',
        'costo': (40, 50)
    },
}

OBJETIVO = 500
nuevas_filas = []

for falla, perfil in PERFIL_FALLAS.items():
    actuales = len(df[df['tipo_falla'] == falla])
    objetivo = OBJETIVO
    if actuales >= objetivo:
        continue

    for _ in range(objetivo - actuales):
        equipo = np.random.choice(perfil['equipos'])
        ruido = np.random.uniform(-0.15, 0.15)
        sintomas = {}
        for s in SINTOMAS:
            prob = perfil['sintomas'].get(s, 0.05) + ruido
            prob = np.clip(prob, 0.01, 0.95)
            sintomas[s] = 1 if np.random.random() < prob else 0

        costo = round(np.random.uniform(*perfil['costo']), 2)
        nuevas_filas.append({
            'id': len(df) + len(nuevas_filas) + 1,
            'tipo_equipo': equipo,
            **sintomas,
            'tipo_falla': falla,
            'reparacion': perfil['reparacion'],
            'costo_aprox': costo,
        })

if nuevas_filas:
    df_nuevo = pd.DataFrame(nuevas_filas)
    df = pd.concat([df, df_nuevo], ignore_index=True)
    df.to_csv('DATASET.csv', index=False, encoding='utf-8-sig')
    print(f"Agregados {len(nuevas_filas)} registros nuevos")
    print(f"Total: {len(df)} registros")
    print("\nNueva distribución:")
    print(df['tipo_falla'].value_counts())
else:
    print("Ya están balanceados")
