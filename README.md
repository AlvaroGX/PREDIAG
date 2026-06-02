# Sistema Inteligente de Prediagnóstico

Aplicación web ML para diagnosticar fallas en equipos tecnológicos (Laptop, PC, Impresora).

## Stack
- **Backend:** Flask + scikit-learn (RandomForest)
- **Frontend:** Bootstrap 5 + vanilla JS
- **Datos:** 500 registros × 18 síntomas binarios, 7 tipos de falla

## Instalación

```bash
pip install -r requirements.txt
python app.py
# Abrir http://localhost:5000
```

## API REST

```bash
curl -X POST http://localhost:5000/api/predecir \
  -H "Content-Type: application/json" \
  -d '{"equipo":"Laptop","enciende":"1","sobrecalienta":"1","muy_lento":"1","reinicia_solo":"1","funciona_sin_bateria":"1"}'
```

## Docker

```bash
docker build -t prediagnostico .
docker run -p 5000:5000 prediagnostico
```

## Tests

```bash
pip install pytest
pytest test_app.py -v
```

## Estructura

```
├── app.py            # Backend principal
├── train.py          # Entrenamiento del modelo
├── templates/        # HTML templates
├── static/           # CSS
├── DATASET.csv       # Dataset
├── modelo.pkl        # Modelo entrenado
├── features.pkl      # Nombres de features
├── encoder_equipo.pkl# Codificador de equipos
├── requirements.txt  # Dependencias
├── Dockerfile        # Contenedor
└── test_app.py       # Tests
```
