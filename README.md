
# 🧠 Mental Health Predictor in Tech

[](https://www.google.com/search?q=https://mental-health-tech-survey.streamlit.app)
[](https://www.python.org/downloads/)

## 📌 Visão Geral

Este projeto é uma solução de Ciência de Dados de alto nível desenvolvida para identificar a probabilidade de colaboradores do setor de tecnologia necessitarem de tratamento de saúde mental. A arquitetura foi desenhada com foco em **Engenharia de Software** e **Estatística Inferencial**.

## 💼 Business Case & ROI

  * **Custo da Inação:** O turnover de um desenvolvedor custa \~1.5x seu salário anual. Um desligamento gera um prejuízo estimado de **R$ 150.000**.
  * **Eficiência:** Com um **Recall de 87%**, ao prevenir apenas 3 desligamentos anuais através de intervenções proativas, a economia supera **R$ 400.000**, pagando o projeto e gerando lucro operacional.
  * **Validação:** O modelo apoia o RH (campanhas direcionadas), Gestores (segurança psicológica) e a Diretoria (KPIs de retenção).

## 📊 Diagnóstico da EDA & Feature Selection

Utilizamos o **V de Cramér** para selecionar as variáveis com real força de associação estatística:

  * **Preditores Críticos:** `work_interfere` (0.36) e `family_history` (0.31).
  * **Insights Organizacionais:** Empresas médias sofrem com o "limbo da comunicação", onde o desconhecimento de benefícios (grupo "Don't Know") paralisa a busca por ajuda.
  * **Cultura:** O medo de retaliação é a barreira nº 1 para a transparência com supervisores.

## 📈 Benchmarking e Performance (Curva ROC/AUC)

O **Gradient Boosting** foi selecionado após uma comparação rigorosa entre modelos:

1.  **Gradient Boosting (Vencedor):** Melhor capacidade de distinção entre classes e subida rápida na Sensibilidade (Eixo Y), essencial para triagens de saúde.
2.  **Logistic Regression:** Desempenho sólido, mas inferior em relações não lineares.
3.  **Random Forest:** Apresentou menor generalização comparado ao Boosting neste dataset.

### Métricas Finais (Threshold: 0.40)

| Métrica | Resultado | Interpretação |
| :--- | :--- | :--- |
| **Recall (Sensibilidade)** | **87%** | **Foco Total:** Identifica 87% de quem realmente precisa de ajuda. |
| **Precisão** | **69%** | Eficiência na gestão de alarmes falsos (Falsos Positivos). |
| **Verdadeiros Positivos** | **110** | Casos reais identificados com sucesso no teste. |
| **Falsos Negativos** | **17** | Apenas 17 casos perdidos (Mínimo histórico). |

## 💡 Por que Gradient Boosting? (Feature Importance)

A análise de importância das variáveis confirma a hierarquia de decisão do modelo:

  * **Fatores Pessoais (Topo):** Histórico Familiar e Interferência no Trabalho dominam a predição.
  * **Políticas Internas (Meio):** Facilidade de licença e opções de cuidado têm peso estratégico para a empresa.
  * **Demografia (Cauda):** Gênero e País demonstraram ser ruído estatístico, provando que a saúde mental é uma questão **humana e universal**.

## 🛠️ Estrutura do Projeto

```text
├── data/               # Raw (original) e Processed (limpo)
├── models/             # final_model_health.pkl (Modelo treinado)
├── notebooks/          # EDA, Curva ROC e V de Cramér
├── src/                # Código Modular (Preprocessing, Training, Eval)
├── main.py             # Orquestrador do Pipeline de Dados
├── app.py              # Interface Web (Streamlit)
└── requirements.txt    # Dependências do projeto
```

## 🚀 Como Executar

1.  Instale as dependências: `pip install -r requirements.txt`
2.  Execute o pipeline de treino e avaliação: `python main.py`
3.  Inicie a interface de predição: `streamlit run app.py`

-----

**Desenvolvido por [Felipe Tamiozzo](https://www.google.com/search?q=https://linkedin.com/in/felipe-tamiozzo)**
*Cientista de Dados focado em converter dados em decisões estratégicas.*

-----

