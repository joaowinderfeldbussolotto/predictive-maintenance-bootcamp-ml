import os
import json
from typing import Dict, Any, List

import requests
import streamlit as st
import pandas as pd


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def api_post(path: str, payload: Dict[str, Any]):
    url = f"{API_BASE_URL}{path}"
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException as e:
        return None, str(e)


def api_get(path: str):
    url = f"{API_BASE_URL}{path}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException as e:
        return None, str(e)


st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")

st.title("🛠️ Predictive Maintenance Dashboard")
st.caption("Simple Streamlit UI integrated with FastAPI")

with st.sidebar:
    st.header("⚙️ Settings")
    api_url_input = st.text_input("API Base URL", value=API_BASE_URL, help="FastAPI base URL")
    if api_url_input and api_url_input != API_BASE_URL:
        API_BASE_URL = api_url_input.rstrip("/")
    st.markdown(f"Current API: `{API_BASE_URL}`")
    health, err = api_get("/health/")
    if health:
        st.success("API OK ✅")
    else:
        st.error(f"API unavailable: {err}")

st.subheader("🔮 Single Prediction")
col1, col2, col3 = st.columns(3)

with col1:
    tipo = st.selectbox("tipo (L/M/H)", options=["L", "M", "H"], index=1)
    temperatura_ar = st.number_input("temperatura_ar (K)", value=298.1, step=0.1)
    temperatura_processo = st.number_input("temperatura_processo (K)", value=308.6, step=0.1)

with col2:
    umidade_relativa = st.number_input("umidade_relativa (%)", value=65.0, step=0.1)
    velocidade_rotacional = st.number_input("velocidade_rotacional (RPM)", value=1551.0, step=1.0)
    torque = st.number_input("torque (Nm)", value=42.8, step=0.1)

with col3:
    desgaste_da_ferramenta = st.number_input("desgaste_da_ferramenta (min)", value=108.0, step=1.0)
    id_val = st.text_input("id (opcional)", value="")
    id_produto = st.text_input("id_produto (opcional)", value="")

# Dois botões para diferentes tipos de predição
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    binary_pred_btn = st.button("🎯 Classificação Binária", type="primary", use_container_width=True)
with col_btn2:
    multi_pred_btn = st.button("🔍 Classificação Multi-Label", type="secondary", use_container_width=True)

# Preparar payload comum
def get_payload():
    payload = {
        "tipo": tipo,
        "temperatura_ar": temperatura_ar,
        "temperatura_processo": temperatura_processo,
        "umidade_relativa": umidade_relativa,
        "velocidade_rotacional": velocidade_rotacional,
        "torque": torque,
        "desgaste_da_ferramenta": desgaste_da_ferramenta,
    }
    if id_val:
        payload["id"] = id_val
    if id_produto:
        payload["id_produto"] = id_produto
    return payload

# Processamento da Classificação Binária
if binary_pred_btn:
    payload = get_payload()
    
    with st.spinner("Executando classificação binária..."):
        data, err = api_post("/predictions/binary-classification", payload)
    
    if err:
        st.error(f"Erro na classificação binária: {err}")
    else:
        st.success("🎯 Resultado da Classificação Binária")
        
        col_result1, col_result2 = st.columns(2)
        with col_result1:
            # Resultado principal
            falha = "SIM" if data["falha_maquina"] else "NÃO"
            color = "red" if data["falha_maquina"] else "green"
            st.markdown(f"### Falha da Máquina: :{color}[{falha}]")
            
            # Métricas
            st.metric("Probabilidade de Falha", f"{data['probabilidade_falha']*100:.1f}%")
            st.metric("Probabilidade Sem Falha", f"{data['probabilidade_sem_falha']*100:.1f}%")
            
        with col_result2:
            # Gráfico de barras das probabilidades
            prob_data = {
                "Falha": data['probabilidade_falha'],
                "Sem Falha": data['probabilidade_sem_falha']
            }
            df_prob = pd.DataFrame(list(prob_data.items()), columns=["Resultado", "Probabilidade"])
            st.bar_chart(df_prob.set_index("Resultado"))
        
        # JSON completo
        with st.expander("📋 Resposta completa da API"):
            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")

# Processamento da Classificação Multi-Label
if multi_pred_btn:
    payload = get_payload()

    with st.spinner("Executando classificação multi-label..."):
        data, err = api_post("/predictions/predict", payload)
    
    if err:
        st.error(f"Erro na classificação multi-label: {err}")
    else:
        st.success("🔍 Resultado da Classificação Multi-Label")
        
        col_result1, col_result2 = st.columns([1, 2])
        
        with col_result1:
            # Resultado principal
            will_fail = "SIM" if data["will_fail"] else "NÃO"
            color = "red" if data["will_fail"] else "green"
            st.markdown(f"### Vai Falhar: :{color}[{will_fail}]")
            
            # Métricas principais
            st.metric("Probabilidade de Falha", f"{data['machine_failure_probability']*100:.1f}%")
            st.metric("Nível de Risco", data["risk_level"].title())
            most_likely = data.get("most_likely_failure")
            if most_likely:
                # Mapeamento dos códigos para nomes mais legíveis
                failure_names = {
                    "FDF": "Desgaste da Ferramenta",
                    "FDC": "Dissipação de Calor", 
                    "FP": "Falha de Potência",
                    "FTE": "Tensão Excessiva",
                    "FA": "Falha Aleatória"
                }
                readable_name = failure_names.get(most_likely, most_likely)
                st.metric("Tipo Mais Provável", readable_name)
            else:
                st.metric("Tipo Mais Provável", "Nenhum")
                
        with col_result2:
            # Gráfico das probabilidades por tipo de falha
            probs = data.get("failure_type_probs", {})
            if probs:
                # Criar DataFrame com nomes mais legíveis
                failure_names = {
                    "FDF": "Desgaste Ferramenta",
                    "FDC": "Dissipação Calor", 
                    "FP": "Falha Potência",
                    "FTE": "Tensão Excessiva",
                    "FA": "Falha Aleatória"
                }
                
                df_probs = pd.DataFrame([
                    {"Tipo de Falha": failure_names.get(k, k), "Probabilidade": v}
                    for k, v in probs.items()
                ])
                
                st.subheader("Probabilidades por Tipo de Falha")
                st.bar_chart(df_probs.set_index("Tipo de Falha"))
                
                # Tabela com valores exatos
                with st.expander("📊 Valores Detalhados"):
                    df_probs["Probabilidade (%)"] = (df_probs["Probabilidade"] * 100).round(2)
                    st.dataframe(df_probs[["Tipo de Falha", "Probabilidade (%)"]], use_container_width=True)
        
        # JSON completo
        with st.expander("📋 Resposta completa da API"):
            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")

st.divider()
st.subheader("📦 Batch Prediction (CSV)")
st.caption("⚠️ Esta funcionalidade está indisponível no momento. Use as classificações individuais por enquanto.")

uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, disabled=True)
batch_btn = st.button("🚀 Processar Lote", disabled=True)

if uploaded is not None or batch_btn:
    st.info("🔧 Funcionalidade em desenvolvimento. Em breve estará disponível!")
    st.markdown("""
    **Recursos planejados:**
    - Upload de arquivo CSV com múltiplas medições
    - Processamento em lote das predições
    - Download dos resultados em formato CSV
    - Resumo estatístico das predições
    """)
