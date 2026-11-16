import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import llm
from dados import ARQ_NOTICIAS, carregar_noticias, carregar_indicadores
from schemas import (
    TextoEntrada,
    RespostaProcessamento,
    ResultadoSentimento,
    ErroResposta,
    StatusModelo,
)

TIMEOUT_SEGUNDOS = 15
ROTULOS_PT = {"positive": "positivo", "neutral": "neutro", "negative": "negativo"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo local uma única vez, ao iniciar a API
    llm.carregar_modelo()
    yield


app = FastAPI(
    title="EquidadeJá Brasil API",
    description="Consulta e envio de dados sobre equidade de gênero no trabalho.",
    version="2.0.0",
    lifespan=lifespan,
)


class Noticia(BaseModel):
    titulo: str
    fonte: str
    data: str
    link: str


def para_json(df: pd.DataFrame):
    """Troca valores vazios (NaN) por None, senão o JSON quebra."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


@app.get("/", tags=["Geral"])
def status():
    """Verifica se a API está no ar."""
    return {"status": "ok", "projeto": "EquidadeJá Brasil"}


@app.get("/indicadores", tags=["Indicadores"])
def listar_indicadores(
    ano_inicio: Optional[int] = Query(None, description="Filtra a partir deste ano"),
    ano_fim: Optional[int] = Query(None, description="Filtra até este ano"),
):
    """Retorna os indicadores de gênero, com filtro opcional por período."""
    df = carregar_indicadores()
    if ano_inicio is not None:
        df = df[df["ano"] >= ano_inicio]
    if ano_fim is not None:
        df = df[df["ano"] <= ano_fim]
    return para_json(df)


@app.get("/noticias", tags=["Notícias"])
def listar_noticias(
    fonte: Optional[str] = Query(None, description="Filtra por fonte (busca parcial)"),
    limite: int = Query(20, ge=1, le=100, description="Máximo de resultados"),
):
    """Retorna as notícias coletadas, com filtro opcional por fonte."""
    df = carregar_noticias()
    if fonte:
        df = df[df["fonte"].str.contains(fonte, case=False, na=False)]
    return para_json(df.head(limite))


@app.post("/noticias", status_code=201, tags=["Notícias"])
def adicionar_noticia(noticia: Noticia):
    """Adiciona uma nova notícia ao arquivo de dados."""
    df = carregar_noticias()
    if noticia.link in df["link"].values:
        raise HTTPException(status_code=409, detail="Notícia já cadastrada.")
    novo = pd.DataFrame([noticia.model_dump()])
    pd.concat([df, novo], ignore_index=True).to_csv(ARQ_NOTICIAS, index=False)
    return {"mensagem": "Notícia adicionada com sucesso", "total": len(df) + 1}


@app.get("/estatisticas", tags=["Geral"])
def estatisticas():
    """Resumo: total de notícias e média de cada indicador."""
    ind = carregar_indicadores()
    return {
        "total_noticias": len(carregar_noticias()),
        "media_por_indicador": ind.groupby("indicador")["valor"].mean().round(2).to_dict(),
    }


# ---------------------- ROTAS NOVAS (TP4 - LLM local) ----------------------

@app.get("/status_modelo", response_model=StatusModelo, tags=["LLM"])
def status_modelo():
    """Informa se o modelo de linguagem local foi carregado."""
    return StatusModelo(
        api="ok",
        modelo=llm.MODELO,
        modelo_carregado=llm.modelo_carregado(),
    )


@app.post(
    "/processar_texto",
    response_model=RespostaProcessamento,
    tags=["LLM"],
    responses={
        400: {"model": ErroResposta, "description": "Texto inválido (só espaços)"},
        500: {"model": ErroResposta, "description": "Erro interno ao processar"},
        503: {"model": ErroResposta, "description": "Modelo LLM indisponível"},
        504: {"model": ErroResposta, "description": "Tempo limite excedido"},
    },
)
async def processar_texto(entrada: TextoEntrada):
    """Analisa o sentimento de um texto usando o modelo local da HuggingFace."""
    if not llm.modelo_carregado():
        raise HTTPException(
            status_code=503,
            detail=f"Modelo LLM indisponível: {llm.erro_carga()}",
        )

    texto = entrada.texto.strip()
    if not texto:
        raise HTTPException(
            status_code=400,
            detail="O texto não pode conter apenas espaços.",
        )

    inicio = time.perf_counter()
    try:
        saida = await asyncio.wait_for(
            asyncio.to_thread(llm.classificar, texto),
            timeout=TIMEOUT_SEGUNDOS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"O processamento excedeu {TIMEOUT_SEGUNDOS} segundos.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar o texto: {e}",
        )

    rotulo = saida["label"].lower()
    return RespostaProcessamento(
        texto_original=texto,
        tipo_analise="analise_de_sentimento",
        modelo=llm.MODELO,
        resultado=ResultadoSentimento(
            rotulo=rotulo,
            rotulo_pt=ROTULOS_PT.get(rotulo, rotulo),
            confianca=round(saida["score"], 4),
        ),
        tempo_processamento_ms=round((time.perf_counter() - inicio) * 1000, 2),
    )