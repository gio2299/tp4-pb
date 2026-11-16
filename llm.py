from transformers import pipeline

MODELO = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

_classificador = None
_erro_carga = None

def carregar_modelo():
    global _classificador, _erro_carga
    try:
        _classificador = pipeline("text-classification", model=MODELO)
        _erro_carga = None
    except Exception as e:
        _classificador = None
        _erro_carga = str(e)

def modelo_carregado() -> bool:
    return _classificador is not None

def erro_carga():
    return _erro_carga

def classificar(texto: str) -> dict:
    return _classificador(texto, truncation=True, max_length=512)[0]