# EquidadeJá Brasil

Painel de equidade de gênero no mercado de trabalho brasileiro.
Projeto de Bloco: Inteligência Artificial Aplicada — Giovanna Silva.

## O que o projeto tem
- Aplicação Streamlit com múltiplas páginas (indicadores, notícias, nuvem de palavras, upload/download, API e sobre)
- API com FastAPI (consulta e envio de dados)
- Web scraping dinâmico com Selenium (texto completo das notícias)
- Corpus de dados preparado para uso com LLMs
- **(TP4)** Análise de sentimento com um modelo de linguagem da HuggingFace executado localmente e integrado à API

## Como instalar
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1    # Windows PowerShell | Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```
O `requirements.txt` inclui `torch` e `transformers`. Na primeira execução da API, o modelo é baixado automaticamente (cerca de 541 MB); depois disso funciona offline, pelo cache local.

## Como rodar
**Aplicação (Streamlit):**
```bash
streamlit run app.py
```

**API (FastAPI)**, em outro terminal:
```bash
python -m uvicorn api:app --reload --port 8000
```
Documentação interativa: http://localhost:8000/docs
(ao iniciar, a API carrega o modelo de linguagem; aguarde `Application startup complete`)

**Scraping com Selenium** (requer Google Chrome instalado):
```bash
python scraping_selenium.py
```

**Preparar o corpus para LLM:**
```bash
python preparar_corpus.py
```

## Rotas da API
| Método | Rota | Função |
|---|---|---|
| GET | `/` | Verifica se a API está no ar |
| GET | `/indicadores` | Lista indicadores (filtros `ano_inicio`, `ano_fim`) |
| GET | `/noticias` | Lista notícias (filtros `fonte`, `limite`) |
| POST | `/noticias` | Cadastra uma notícia |
| GET | `/estatisticas` | Resumo dos dados |
| GET | `/status_modelo` | **(TP4)** Informa se o modelo local está carregado |
| POST | `/processar_texto` | **(TP4)** Analisa o sentimento de um texto com o modelo local |

---

## TP4: Integração com LLM local (HuggingFace + FastAPI)

A API analisa o sentimento de textos (por exemplo, manchetes sobre equidade de gênero) com um modelo de linguagem que roda **localmente**, sem nuvem e sem chave de API.

### Modelo escolhido
`lxyuan/distilbert-base-multilingual-cased-sentiments-student` (HuggingFace Hub)
- Tarefa: classificação de sentimento (positivo, neutro, negativo), com suporte a português (12 idiomas)
- Origem: modelo destilado (professor: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`; aluno: `distilbert-base-multilingual-cased`), treinado com o dataset `tyqiangz/multilingual-sentiments`
- Tamanho: cerca de 0,1 bilhão de parâmetros; arquivo de pesos (`model.safetensors`) de 541 MB
- Licença: Apache 2.0
- Execução em CPU, tempo medido de 20 a 35 ms por texto

### Ambiente de teste
- Windows 10 Home Single Language, versão 22H2 (build 19045.6466), 64 bits, Python 3.14.7
- Intel Core i7-9750H 2.60 GHz, 16 GB de RAM (GPU NVIDIA RTX 2060 presente, mas não utilizada)
- torch 2.15.0+cpu, transformers 5.23.0

### Como a integração foi feita
1. `llm.py` carrega o modelo com `pipeline("text-classification")` da biblioteca Transformers e expõe a função `classificar`.
2. `schemas.py` define os Response Models (Pydantic).
3. `api.py` carrega o modelo uma única vez no `lifespan` do FastAPI e expõe as rotas `/status_modelo` e `/processar_texto`.
4. Ajustes: `truncation=True` e `max_length=512` (textos longos), `asyncio.to_thread` e `asyncio.wait_for` (a inferência não trava a API e tem limite de 15 s) e PyTorch versão CPU (dispensa GPU).

### Exemplo de uso
Requisição `POST /processar_texto`:
```json
{"texto": "Brasil avança na igualdade salarial e mais mulheres chegam à liderança"}
```
Resposta (200):
```json
{
  "texto_original": "Brasil avança na igualdade salarial e mais mulheres chegam à liderança",
  "tipo_analise": "analise_de_sentimento",
  "modelo": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
  "resultado": {"rotulo": "positive", "rotulo_pt": "positivo", "confianca": 0.7829},
  "tempo_processamento_ms": 21.34
}
```

### Tratamento de erros
| Código | Cenário | Origem |
|---|---|---|
| 400 | Texto contém apenas espaços | HTTPException na rota |
| 409 | Notícia já cadastrada (`POST /noticias`) | HTTPException na rota |
| 422 | Texto vazio, curto (< 3) ou longo (> 5000 caracteres) | Validação Pydantic |
| 500 | Erro inesperado durante a inferência | HTTPException na rota |
| 503 | Modelo não carregado | HTTPException na rota |
| 504 | Processamento excedeu 15 s | HTTPException na rota |

### Arquivos novos do TP4
- `llm.py`: carrega o modelo e classifica textos
- `schemas.py`: Response Models
- `api.py`: rotas atualizadas, com `lifespan`

### Problemas encontrados na instalação
- O índice do PyTorch CPU (`--index-url`) não contém o `transformers`; os pacotes foram instalados em comandos separados.
- A pasta `.venv` copiada do TP3 apontava para o caminho antigo; foi recriada e passou-se a usar `python -m pip`.
- O aviso do `huggingface_hub` sobre *symlinks* no Windows não afeta o funcionamento (apenas usa mais disco no cache).

### Limitações
- O modelo erra mais em frases mistas (ex.: "presença feminina cresce, mas desigualdades persistem" saiu como negativo com 50,8% de confiança). Confiança baixa indica incerteza.
- Os resultados são sugestões e devem ser revisados.

## Documentação
Veja a pasta `Docs/` (Project Charter e Data Summary Report).

## Uso de IA
Ferramentas de IA generativa (Claude, Anthropic) foram usadas como apoio no código e na documentação, com revisão da autora. Em tempo de execução, a aplicação **não** depende de nenhuma API de IA em nuvem: o modelo de linguagem é local (HuggingFace).