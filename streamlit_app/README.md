# Streamlit App - Predictive Maintenance

Esta pasta contém a interface web interativa desenvolvida com Streamlit para uso dos modelos de manutenção preditiva.

## Estrutura
- `app.py`: Aplicação principal Streamlit
- `requirements.txt`: Dependências Python
- `Dockerfile`: Container do Streamlit

## Como usar

```bash
# Instale as dependências
pip install -r requirements.txt

# Rode o Streamlit
streamlit run app.py --server.port 8501
```

## Funcionalidades
- Formulário para entrada dos dados dos sensores
- Botão para classificação binária (funcional)
- Botão para classificação multi-label (em construção)
- Visualização dos resultados e probabilidades

## Observações
- Apenas o endpoint binário está funcional; multi-label retorna mensagem de construção.
- A interface conecta automaticamente à API FastAPI.
