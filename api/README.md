# API - Predictive Maintenance

Esta pasta contém a API REST desenvolvida com FastAPI para manutenção preditiva de máquinas industriais.

## Estrutura
- `main.py`: Inicialização da API e configuração dos endpoints
- `app/routes/`: Endpoints de saúde, predição e modelos
- `app/services/`: Lógica de negócio e carregamento dos modelos
- `app/schemas/`: Schemas Pydantic para validação dos dados
- `app/utils/`: Configuração, logger e utilitários
- `ml_models/`: Modelos treinados (.pkl, .joblib)
- `requirements.txt`: Dependências Python
- `Dockerfile`: Container da API

## Como usar

```bash
# Instale as dependências
pip install -r requirements.txt

# Rode a API
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints principais
- `/predictions/binary-classification`: Classificação binária (✅ funcional)
- `/predictions/predict`: Classificação multi-label (✅ funcional)
- `/predictions/predict/batch`: Predições em lote (🚧 indisponível)
- `/health/`: Health check
- `/models/info`: Informações do modelo

## Status dos Modelos
- **Classificação Binária**: ✅ Totalmente funcional com modelo XGBoost
- **Classificação Multi-label**: ✅ Funcional com pipeline completo (preprocessamento + modelo)
- **Processamento em Lote**: 🚧 Funcionalidade indisponível no momento
