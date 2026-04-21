import streamlit as st
import pandas as pd
import joblib

# Configuração da página
st.set_page_config(page_title="Preditor de Saúde Mental", page_icon="🧠")

st.title("🧠 Assistente de Triagem de Saúde Mental")
st.markdown("""
Este aplicativo utiliza um modelo de Machine Learning (Gradient Boosting) para identificar 
a probabilidade de um colaborador precisar de suporte especializado.
""")

# 1. Carregar o modelo salvo
@st.cache_resource # Mantém o modelo na memória para ser rápido
def load_model():
    return joblib.load("models/final_model_health.pkl")

model_pipeline = load_model()

# 2. Criar o formulário de entrada
st.sidebar.header("Dados do Colaborador")

def user_input_features():
    work_interfere = st.sidebar.selectbox("Interferência no Trabalho", ("Never", "Rarely", "Sometimes", "Often"))
    family_history = st.sidebar.selectbox("Histórico Familiar", ("No", "Yes"))
    leave = st.sidebar.selectbox("Facilidade de Licença", ("Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"))
    benefits = st.sidebar.selectbox("Empresa oferece Benefícios?", ("Yes", "No", "Don't know"))
    care_options = st.sidebar.selectbox("Sabe das Opções de Cuidado?", ("Yes", "No", "Not sure"))
    anonymity = st.sidebar.selectbox("Anonimato Protegido?", ("Yes", "No", "Don't know"))
    gender = st.sidebar.selectbox("Gênero", ("Male", "Female", "Other"))
    
    # Criar um DataFrame com as colunas que o modelo espera
    data = {
        'work_interfere': work_interfere,
        'family_history': family_history,
        'leave': leave,
        'benefits': benefits,
        'care_options': care_options,
        'anonymity': anonymity,
        'Gender': gender,
        # Adicione as outras colunas que você usou no treino com valores padrão se necessário
        'Country': 'United States', 
        'phys_health_consequence': 'No',
        'mental_health_interview': 'No',
        'no_employees': '26-100'
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 3. Predição
if st.button("Analisar Risco"):
    # Obter probabilidade
    proba = model_pipeline.predict_proba(input_df)[0, 1]
    
    st.subheader("Resultado da Análise")
    if proba >= 0.40:
        st.error(f"Probabilidade de Risco: {proba:.2%}")
        st.warning("Recomendação: O colaborador possui indicadores que sugerem a necessidade de suporte profissional.")
    else:
        st.success(f"Probabilidade de Risco: {proba:.2%}")
        st.info("Recomendação: O colaborador apresenta baixo risco atual.")

st.markdown("---")
st.caption("Desenvolvido por Felipe Tamiozzo - Cientista de Dados")