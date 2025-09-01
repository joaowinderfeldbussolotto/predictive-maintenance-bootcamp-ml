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
    multi_pred_btn = st.button("🔍 Classificação Multi-Label", use_container_width=True)

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

# Processamento da Classificação Multi-Label (em construção)
if multi_pred_btn:
    payload = get_payload()

    with st.spinner("Verificando status do endpoint multi-label..."):
        data, err = api_post("/predictions/predict", payload)
    
    if err:
        st.error(f"Erro na classificação multi-label: {err}")
    else:
        # Verificar se é mensagem de construção
        if "message" in data and "construção" in data["message"]:
            st.warning(f"⚠️ {data.get('message', 'Endpoint em construção')}")
            st.info(data.get('detail', 'Este endpoint está sendo desenvolvido.'))
            
            if "available_endpoint" in data:
                st.markdown(f"**Endpoint disponível:** `{data['available_endpoint']}`")
                st.markdown("👆 Use o botão **Classificação Binária** acima para predições funcionais!")
                
        else:
            # Caso seja resposta normal (fallback)
            st.success("🔍 Resultado da Classificação Multi-Label")
            
            left, right = st.columns([2, 3])
            with left:
                st.metric("Probabilidade de Falha", f"{data['machine_failure_probability']*100:.1f}%")
                will_fail = "SIM" if data["will_fail"] else "NÃO"
                st.metric("Vai Falhar", will_fail)
                st.metric("Nível de Risco", data["risk_level"].title())
                st.metric("Tipo de Falha Mais Provável", data["most_likely_failure"] or "-")
                
            with right:
                probs = data.get("failure_type_probs", {})
                if probs:
                    df = pd.DataFrame({"failure_type": list(probs.keys()), "prob": list(probs.values())})
                    st.bar_chart(df.set_index("failure_type"))
        
        # JSON completo sempre mostrado
        with st.expander("📋 Resposta completa da API"):
            st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")

st.divider()
st.subheader("📦 Batch Prediction (CSV) - Em Construção")
st.caption("⚠️ Esta funcionalidade está em desenvolvimento. Use a classificação binária individual por enquanto.")

uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, disabled=True)
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Predict batch", disabled=True):
            records: List[Dict[str, Any]] = df.to_dict(orient="records")
            with st.spinner("Verificando endpoint batch..."):
                data, err = api_post("/predictions/predict/batch", {"measurements": records})
            if err:
                st.error(f"Batch request failed: {err}")
            else:
                # Verificar se é mensagem de construção
                if "message" in data and "construção" in data["message"]:
                    st.warning(f"⚠️ {data.get('message', 'Endpoint em construção')}")
                    st.info(data.get('detail', 'Este endpoint está sendo desenvolvido.'))
                    
                    if "available_endpoint" in data:
                        st.markdown(f"**Endpoint disponível:** `{data['available_endpoint']}`")
                        
                else:
                    st.success("Batch prediction completed")
                    st.json(data.get("summary", {}))
                    # Show first 10 predictions
                    preds = data.get("predictions", [])
                    if preds:
                        dfp = pd.DataFrame(preds)
                        st.dataframe(dfp.head(10))
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
