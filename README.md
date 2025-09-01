# 🏭 Predictive Maintenance System - ML Bootcamp

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.114.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.0.4-yellow)

**Sistema Inteligente de Manutenção Preditiva para Máquinas Industriais**

Este projeto implementa um sistema completo de manutenção preditiva usando Machine Learning, desenvolvido como parte do bootcamp de ML. O sistema é capaz de prever falhas em máquinas industriais com **classificação binária** e **multi-label** baseado em dados de sensores IoT.

## 🎯 Objetivo do Projeto

Desenvolver um sistema inteligente de manutenção preditiva que:

- ✅ **Prevê falhas binárias** (vai falhar/não vai falhar) com alta precisão
- 🚧 **Identifica tipos de falhas** (em desenvolvimento - FDF, FDC, FP, FTE, FA)
- ✅ **Calcula probabilidades** e níveis de risco
- ✅ **Interface web interativa** com Streamlit
- ✅ **API REST completa** para integração em sistemas existentes
- ✅ **Containerização** com Docker para deploy fácil

## 🏗️ Arquitetura do Sistema

```
predictive-maintenance-bootcamp-ml/
├── 🚀 api/                      # FastAPI REST API
│   ├── app/                     # Código da aplicação
│   │   ├── routes/              # Endpoints (health, predictions, models)
│   │   ├── services/            # Lógica de negócio (ModelService)
│   │   ├── schemas/             # Modelos Pydantic (validation)
│   │   └── utils/               # Configuração e utilitários
│   ├── ml_models/               # Modelos treinados (.pkl, .joblib)
│   │   ├── binary_classification.pkl    # ✅ Modelo binário (funcional)
│   │   ├── pipeline_multilabel.pkl      # 🚧 Modelo multilabel
│   │   ├── decision_tree_tuned_model.joblib
│   │   ├── preprocessor.joblib
│   │   └── xgboost_undersample_pipeline.pkl
│   ├── saved_models/            # Modelos adicionais
│   ├── Dockerfile               # Container da API
│   └── requirements.txt         # Dependências Python
├── 🖥️ streamlit_app/            # Interface Web
│   ├── app.py                   # Aplicação Streamlit
│   ├── Dockerfile               # Container do Streamlit
│   └── requirements.txt         # Dependências
├── 🐳 docker-compose.yml        # Orquestração de containers
├── 📝 logs/                     # Logs da aplicação
├── 🚀 start_api.sh              # Script de inicialização
└── 📖 README.md                 # Documentação
```

## 📊 Dados e Problema

### Atributos de Entrada (Sensores IoT)

| Campo | Tipo | Descrição | Range |
|-------|------|-----------|-------|
| `tipo` | Enum | Tipo da máquina (L/M/H) | L, M, H |
| `temperatura_ar` | Float | Temperatura do ar ambiente (K) | 290-310 |
| `temperatura_processo` | Float | Temperatura do processo (K) | 300-320 |
| `umidade_relativa` | Float | Umidade relativa do ar (%) | 0-100 |
| `velocidade_rotacional` | Float | Velocidade em RPM | 1000-3000 |
| `torque` | Float | Torque da máquina (Nm) | 0-100 |
| `desgaste_da_ferramenta` | Float | Tempo de uso da ferramenta (min) | 0-300 |

### Tipos de Falha (Multi-label)

- 🔧 **FDF**: Falha por Desgaste da Ferramenta
- 🌡️ **FDC**: Falha por Dissipação de Calor  
- ⚡ **FP**: Falha de Potência
- 💪 **FTE**: Falha por Tensão Excessiva
- 🎲 **FA**: Falha Aleatória

## 🚀 Quick Start

### 1. Usando Docker (Recomendado)

```bash
# Clone o repositório
git clone https://github.com/joaowinderfeldbussolotto/predictive-maintenance-bootcamp-ml.git
cd predictive-maintenance-bootcamp-ml

# Execute o script de inicialização
chmod +x start_api.sh
./start_api.sh
```

### 2. Instalação Manual

```bash
# API
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Streamlit (terminal separado)
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

### 3. Acesso às Aplicações

Após a inicialização:

- 🌐 **Interface Web (Streamlit)**: http://localhost:8501
- 📖 **Documentação da API**: http://localhost:8000/docs
- 🏥 **Health Check**: http://localhost:8000/health
- ℹ️ **API Info**: http://localhost:8000/info

## 🔍 Usando a API

### 1. Classificação Binária (✅ Funcional)

```python
import requests

# Dados do sensor
data = {
    "tipo": "M",
    "temperatura_ar": 298.1,
    "temperatura_processo": 308.6,
    "umidade_relativa": 65.0,
    "velocidade_rotacional": 1551.0,
    "torque": 42.8,
    "desgaste_da_ferramenta": 108.0,
    "id": "sensor-001",
    "id_produto": "M-1001"
}

# Predição binária (vai falhar ou não)
response = requests.post(
    "http://localhost:8000/predictions/binary-classification", 
    json=data
)
result = response.json()

print(f"Falha da Máquina: {result['falha_maquina']}")
print(f"Probabilidade de Falha: {result['probabilidade_falha']:.2%}")
```

**Resposta:**
```json
{
    "falha_maquina": false,
    "probabilidade_falha": 0.12,
    "probabilidade_sem_falha": 0.88,
    "id": "sensor-001",
    "id_produto": "M-1001"
}
```

### 2. Classificação Multi-label (🚧 Em Desenvolvimento)

```python
# Multi-label classification (em construção)
response = requests.post(
    "http://localhost:8000/predictions/predict", 
    json=data
)
# Retorna: "Endpoint em construção. Use /binary-classification"
```

### 3. Exemplo com cURL

```bash
# Teste rápido
curl -X POST "http://localhost:8000/predictions/binary-classification" \
  -H "Content-Type: application/json" \
  -d '{
    "tipo": "M",
    "temperatura_ar": 298.1,
    "temperatura_processo": 308.6,
    "umidade_relativa": 65.0,
    "velocidade_rotacional": 1551.0,
    "torque": 42.8,
    "desgaste_da_ferramenta": 108.0
  }'
```

## 🖥️ Interface Web (Streamlit)

A interface web oferece:

### ✅ Classificação Binária
- Formulário interativo para entrada de dados
- Predição em tempo real
- Visualização de probabilidades
- Histórico de predições

### 🚧 Classificação Multi-label  
- Mensagem de "Em construção"
- Redirecionamento para classificação binária

### 📊 Features
- Validação de entrada em tempo real
- Feedback visual das predições
- Layout responsivo e intuitivo
- Conexão automática com a API

## 📊 Endpoints da API

### 🎯 Predições

| Método | Endpoint | Status | Descrição |
|--------|----------|--------|-----------|
| `POST` | `/predictions/binary-classification` | ✅ Funcional | Classificação binária |
| `POST` | `/predictions/predict` | 🚧 Em construção | Multi-label (mockado) |
| `POST` | `/predictions/predict/batch` | 🚧 Em construção | Predições em lote |
| `GET` | `/predictions/example` | ✅ Funcional | Exemplo de payload |

### 🤖 Modelos

| Método | Endpoint | Status | Descrição |
|--------|----------|--------|-----------|
| `GET` | `/models/info` | ✅ Funcional | Informações do modelo |
| `GET` | `/models/status` | ✅ Funcional | Status de carregamento |
| `GET` | `/models/features` | ✅ Funcional | Features esperadas |
| `POST` | `/models/reload` | ✅ Funcional | Recarregar modelos |

### 🏥 Health & Monitoring

| Método | Endpoint | Status | Descrição |
|--------|----------|--------|-----------|
| `GET` | `/health/` | ✅ Funcional | Status completo |
| `GET` | `/health/liveness` | ✅ Funcional | Probe de vida |
| `GET` | `/health/readiness` | ✅ Funcional | Probe de prontidão |
| `GET` | `/info` | ✅ Funcional | Informações gerais |

## 🧪 Testando o Sistema

### Teste Rápido da API

```bash
# Health check
curl http://localhost:8000/health/

# Exemplo de predição
curl http://localhost:8000/predictions/example

# Predição binária
curl -X POST http://localhost:8000/predictions/binary-classification \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

### Casos de Teste

**Caso 1: Máquina Saudável**
```json
{
    "tipo": "L",
    "temperatura_ar": 295.0,
    "temperatura_processo": 305.0,
    "umidade_relativa": 50.0,
    "velocidade_rotacional": 1200.0,
    "torque": 30.0,
    "desgaste_da_ferramenta": 50.0
}
```

**Caso 2: Máquina com Risco**
```json
{
    "tipo": "H",
    "temperatura_ar": 308.0,
    "temperatura_processo": 318.0,
    "umidade_relativa": 85.0,
    "velocidade_rotacional": 2800.0,
    "torque": 85.0,
    "desgaste_da_ferramenta": 250.0
}
```

## 🐳 Deployment com Docker

### Desenvolvimento (API + Streamlit)

```bash
# Subir todos os serviços
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar serviços
docker-compose down
```

### Configuração do Docker Compose

```yaml
services:
  predictive-maintenance-api:
    build: ./api
    ports:
      - "8000:8000"
    volumes:
      - ./api/ml_models:/app/ml_models:ro
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/liveness"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit-app:
    build: ./streamlit_app
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://predictive-maintenance-api:8000
    depends_on:
      - predictive-maintenance-api
```

## 🔧 Configuração e Personalização

### Variáveis de Ambiente

**API (api/.env)**
```bash
# Server
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Model
MODEL_DIR=/app/ml_models
PREDICTION_THRESHOLD=0.5
MAX_BATCH_SIZE=1000
```

**Streamlit (streamlit_app/.env)**
```bash
API_BASE_URL=http://localhost:8000
```

### Modelos Disponíveis

| Arquivo | Tipo | Status | Descrição |
|---------|------|--------|-----------|
| `binary_classification.pkl` | Pipeline | ✅ Funcional | XGBoost + preprocessamento |
| `pipeline_multilabel.pkl` | Pipeline | 🚧 Desenvolvimento | Multi-label classifier |
| `decision_tree_tuned_model.joblib` | Modelo | 🚧 Integração | Decision Tree otimizada |
| `preprocessor.joblib` | Transformer | 🚧 Integração | Preprocessamento standalone |
| `xgboost_undersample_pipeline.pkl` | Pipeline | 🚧 Integração | XGBoost com undersampling |

## 🛠️ Desenvolvimento

### Estrutura do Código

```
api/app/
├── routes/
│   ├── health.py        # Health checks
│   ├── predictions.py   # Endpoints de predição
│   └── models.py        # Gerenciamento de modelos
├── services/
│   └── model_service.py # Lógica de ML
├── schemas/
│   ├── common.py        # Tipos comuns
│   ├── prediction.py    # Schemas de predição
│   └── model.py         # Schemas de modelo
└── utils/
    ├── config.py        # Configurações
    └── logger.py        # Logging
```

### Fluxo de Predição

1. **Entrada**: Dados do sensor via API
2. **Validação**: Pydantic schemas
3. **Preprocessamento**: Adição de `sensor_ok = temperatura_ar > 0`
4. **Predição**: Pipeline carregado (`binary_classification.pkl`)
5. **Resposta**: JSON estruturado

### Adicionando Novos Modelos

1. **Salvar modelo**: Colocar em `api/ml_models/`
2. **Atualizar service**: Adicionar carregamento em `ModelService`
3. **Criar endpoint**: Nova rota em `app/routes/predictions.py`
4. **Documentar**: Atualizar schema e docs

## 📈 Monitoramento e Logs

### Health Checks

```bash
# Status geral
curl http://localhost:8000/health/

# Kubernetes liveness
curl http://localhost:8000/health/liveness

# Kubernetes readiness  
curl http://localhost:8000/health/readiness
```

### Logs Estruturados

```bash
# Ver logs em tempo real
docker-compose logs -f predictive-maintenance-api

# Filtrar por nível
docker-compose logs predictive-maintenance-api | grep ERROR

# Logs do Streamlit
docker-compose logs -f streamlit-app
```

### Métricas de Performance

- **Tempo de Resposta**: < 100ms para predições individuais
- **Throughput**: Suporte a múltiplas requisições simultâneas
- **Disponibilidade**: Health checks automáticos
- **Precisão**: Modelo binário com alta acurácia

## 🚨 Troubleshooting

### Problemas Comuns

**1. Modelo não carrega**
```bash
# Verificar se arquivo existe
ls -la api/ml_models/binary_classification.pkl

# Verificar logs de carregamento
docker-compose logs predictive-maintenance-api | grep "Loading"
```

**2. Porta ocupada**
```bash
# Verificar processos
lsof -i :8000
lsof -i :8501

# Alterar portas no docker-compose.yml
```

**3. Dependências em falta**
```bash
# Reconstruir containers
docker-compose down
docker-compose up --build
```

**4. Erro de conexão Streamlit -> API**
```bash
# Verificar network do Docker
docker network ls
docker-compose logs streamlit-app
```

### Debug Mode

```bash
# Ativar debug na API
echo "DEBUG=true" >> api/.env
docker-compose restart predictive-maintenance-api

# Conectar ao container
docker-compose exec predictive-maintenance-api bash
```

## 📚 Documentação das Pastas Core

### [`api/`](./api/README.md)
API REST para manutenção preditiva. Endpoints, estrutura, uso e observações sobre os modelos e funcionalidades disponíveis.

### [`notebook/`](./notebook/README.md)
Notebooks Jupyter para análise exploratória, modelagem, treinamento e validação dos modelos. Explica o fluxo de ciência de dados e como os artefatos são gerados.

### [`streamlit_app/`](./streamlit_app/README.md)
Interface web interativa para uso dos modelos. Detalha funcionalidades, estrutura e como rodar o app para predição binária e multi-label.

---

> Consulte os READMEs de cada pasta para detalhes técnicos, instruções de uso e observações específicas de cada componente do sistema.

## 📞 Suporte e Contato

Para dúvidas, sugestões ou suporte:

1. 📖 **Documentação**: Consulte `/docs` na API
2. 🏥 **Health Status**: Verifique `/health` 
3. 📝 **Logs**: Analise os logs dos containers
4. 🐛 **Issues**: Abra uma issue no GitHub
5. 💬 **Discussões**: Use as discussões do repositório

