
# 🧠 Mental Health Predictor in Tech

## 📌 Visão Geral

Este projeto é uma solução de Ciência de Dados de alto nível desenvolvida para identificar a probabilidade de colaboradores do setor de tecnologia necessitarem de tratamento de saúde mental. A arquitetura foi desenhada com foco em **Engenharia de Software** e **Estatística Inferencial**.

## 💼 Business Case & ROI

  * **Custo da Inação:** O turnover de um dev custa \~1.5x seu salário anual. Um desligamento gera prejuízo estimado de **R$ 150.000**.
  * **Eficiência:** Com **Recall de 87%**, ao prevenir apenas 3 desligamentos anuais, a economia supera **R$ 400.000**, pagando o projeto e gerando lucro operacional.
  * **Validação:** Modelo apoia o RH (campanhas direcionadas), Gestores (segurança psicológica) e Diretoria (KPIs de retenção).

## 📊 Diagnóstico da EDA & Feature Selection

Utilizamos o **V de Cramér** para selecionar variáveis com real força de associação:

  * **Preditores Críticos:** `work_interfere` (0.36) e `family_history` (0.31).
  * **Insights Organizacionais:** Empresas médias sofrem com o "limbo da comunicação", onde o desconhecimento de benefícios (grupo "Don't Know") paralisa a busca por ajuda.
  * **Cultura:** O medo de retaliação é a barreira nº 1 para a transparência com supervisores.

## 📈 Benchmarking e Performance (Curva ROC/AUC)

O **Gradient Boosting** foi selecionado após comparação rigorosa:

1.  **Gradient Boosting (Vencedor):** Melhor capacidade de distinção entre classes e subida rápida na Sensibilidade (Eixo Y), essencial para diagnósticos de saúde.
2.  **Logistic Regression:** Desempenho sólido, mas inferior em relações não lineares.
3.  **Random Forest:** Apresentou menor generalização comparado ao Boosting neste dataset.

### Métricas Finais (Threshold: 0.40)

| Métrica | Resultado | Interpretação |
| :--- | :--- | :--- |
| **Recall** | **87%** | **Foco Total:** Identifica 87% de quem realmente precisa de ajuda. |
| **Precisão** | **69%** | Eficiência na gestão de alarmes falsos. |
| **Verdadeiros Positivos** | **110** | Casos reais identificados com sucesso. |
| **Falsos Negativos** | **17** | Apenas 17 casos perdidos (Mínimo histórico). |

## 💡 Por que Gradient Boosting? (Feature Importance)

A análise de importância das variáveis confirma a hierarquia do modelo:

  * **Fatores Pessoais (Topo):** Histórico Familiar e Interferência no Trabalho dominam a predição.
  * **Políticas Internas (Meio):** Facilidade de licença e opções de cuidado têm peso estratégico.
  * **Demografia (Cauda):** Gênero e País demonstraram ser ruído estatístico, provando que a saúde mental é uma questão **humana e universal**.

## 🛠️ Estrutura do Projeto

```text
├── data/               # Raw (original) e Processed (limpo)
├── models/             # final_model_health.pkl
├── notebooks/          # EDA, Curva ROC e V de Cramér
├── src/                # Código Modular (Preprocessing, Training, Eval)
├── main.py             # Orquestrador do Pipeline
├── app.py              # Interface Streamlit
└── requirements.txt    # Dependências (Python 3.11)
```

## 🚀 Como Executar

1.  `pip install -r requirements.txt`
2.  `python main.py` (Treina, avalia e salva o modelo)
3.  `streamlit run app.py` (Inicia a interface de predição)

-----

**Desenvolvido por [Felipe Tamiozzo](https://www.google.com/search?q=https://linkedin.com/in/felipe-tamiozzo)**
*Cientista de Dados focado em converter dados em decisões estratégicas.*
