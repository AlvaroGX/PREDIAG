import pytest
import json
from app import app


@pytest.fixture
def client():
    with app.test_client() as c:
        yield c


def sintomas_base(equipo='Laptop'):
    return {
        'equipo': equipo,
        'enciende': '0', 'sufrio_golpe': '0', 'hace_ruido': '0', 'sobrecalienta': '0',
        'no_enciende_del_todo': '0', 'reinicia_solo': '0', 'error_sistema': '0', 'pitidos_encendido': '0',
        'muy_lento': '0', 'pantalla_negra': '0', 'pantalla_azul': '0',
        'funciona_sin_bateria': '0', 'funciona_tras_golpe': '0', 'muy_lento_post_formato': '0',
        'no_imprime': '0', 'atasca_papel': '0', 'imprime_lineas': '0', 'succion_bomba': '0'
    }


class TestIndex:
    def test_get_index(self, client):
        r = client.get('/')
        assert r.status_code == 200
        assert b'Prediagn' in r.data


class TestPrediccion:
    def test_laptop_sobrecalienta(self, client):
        data = sintomas_base('Laptop')
        data.update({'sobrecalienta': '1', 'muy_lento': '1', 'reinicia_solo': '1', 'funciona_sin_bateria': '1'})
        r = client.post('/predecir', data=data)
        assert r.status_code == 200
        assert b'Sobrecalentamiento' in r.data

    def test_pc_golpe(self, client):
        data = sintomas_base('PC de escritorio')
        data.update({'sufrio_golpe': '1', 'hace_ruido': '1', 'no_enciende_del_todo': '1', 'pitidos_encendido': '1'})
        r = client.post('/predecir', data=data)
        assert r.status_code == 200
        assert b'Falla de hardware' in r.data

    def test_impresora(self, client):
        data = sintomas_base('Impresora Epson')
        data.update({'hace_ruido': '1', 'no_imprime': '1', 'atasca_papel': '1', 'imprime_lineas': '1'})
        r = client.post('/predecir', data=data)
        assert r.status_code == 200
        assert b'Falla de impresi' in r.data

    def test_muestra_importancias(self, client):
        data = sintomas_base('Laptop')
        data.update({'sobrecalienta': '1', 'muy_lento': '1', 'reinicia_solo': '1'})
        r = client.post('/predecir', data=data)
        assert r.status_code == 200
        assert b'badge' in r.data

    def test_muestra_confianza(self, client):
        data = sintomas_base('Laptop')
        data.update({'sobrecalienta': '1'})
        r = client.post('/predecir', data=data)
        assert r.status_code == 200
        assert b'progress-bar' in r.data


class TestErrores:
    def test_equipo_invalido(self, client):
        data = sintomas_base('XBox')
        r = client.post('/predecir', data=data)
        assert b'alert-danger' in r.data

    def test_campos_faltantes(self, client):
        r = client.post('/predecir', data={'equipo': 'Laptop'})
        assert b'alert-danger' in r.data


class TestAPI:
    def test_api_json(self, client):
        data = sintomas_base('Laptop')
        data.update({'sobrecalienta': '1', 'muy_lento': '1'})
        r = client.post('/api/predecir', data=json.dumps(data), content_type='application/json')
        assert r.status_code == 200
        resp = r.get_json()
        assert resp['tipo_falla'] == 'Sobrecalentamiento'
        assert 'confianza' in resp
        assert len(resp['importancias']) > 0

    def test_api_equipo_invalido(self, client):
        r = client.post('/api/predecir', json={'equipo': 'XBox'})
        assert r.status_code == 400
        assert 'error' in r.get_json()

    def test_api_costo(self, client):
        data = sintomas_base('Impresora Epson')
        data.update({'no_imprime': '1', 'atasca_papel': '1'})
        r = client.post('/api/predecir', data=json.dumps(data), content_type='application/json')
        resp = r.get_json()
        assert resp['costo'] is not None


class TestHistorial:
    def test_historial_vacio(self, client):
        r = client.get('/historial')
        assert r.status_code == 200

    def test_historial_tras_prediccion(self, client):
        data = sintomas_base('Laptop')
        client.post('/predecir', data=data)
        r = client.get('/historial')
        assert r.status_code == 200
        assert b'Laptop' in r.data


class TestExportar:
    def test_exportar(self, client):
        r = client.post('/exportar', data={
            'tipo_falla': 'Sobrecalentamiento',
            'reparacion': 'Limpieza interna',
            'costo': '$45.00',
            'confianza': '95.2',
            'equipo': 'Laptop'
        })
        assert r.status_code == 200
        assert r.content_type.startswith('text/plain')
        assert b'DIAGN' in r.data
