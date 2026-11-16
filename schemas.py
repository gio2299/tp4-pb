from typing import Literal
from pydantic import BaseModel, Field

class TextoEntrada(BaseModel):
    texto: str = Field(..., min_length=3, max_length=5000,
                       examples=["Brasil avança pouco na igualdade salarial entre mulheres e homens"])

class ResultadoSentimento(BaseModel):
    rotulo: Literal["positive", "neutral", "negative"]
    rotulo_pt: str
    confianca: float = Field(..., ge=0, le=1)

class RespostaProcessamento(BaseModel):
    texto_original: str
    tipo_analise: Literal["analise_de_sentimento"]
    modelo: str
    resultado: ResultadoSentimento
    tempo_processamento_ms: float

class ErroResposta(BaseModel):
    detail: str

class StatusModelo(BaseModel):
    api: str
    modelo: str
    modelo_carregado: bool