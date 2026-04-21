
````markdown
# 🧠 Mental Health Predictor in Tech

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://mental-health-tech-survey.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-EDA_&_Modeling-F37626.svg)](notebooks/mental_tech_eda.ipynb)

## 📌 Visão Geral

Este projeto é uma solução de Ciência de Dados End-to-End desenvolvida para prever a probabilidade de colaboradores do setor de tecnologia necessitarem de tratamento de saúde mental. A arquitetura foi desenhada com foco em **Engenharia de Software**, **Estatística Inferencial** e entrega de **Valor de Negócio**.

## 💼 Business Case & Retorno sobre Investimento (ROI)

* **Custo da Inação:** O turnover de um desenvolvedor pleno/sênior custa em média ~1.5x seu salário anual. Um desligamento não previsto gera um prejuízo estimado de **R$ 150.000**.
* **Eficiência Operacional:** Calibrado com um **Recall de 87%**, o modelo atua como um sistema de aviso antecipado. Ao prevenir apenas 3 desligamentos anuais através de intervenções proativas, a economia financeira supera **R$ 400.000**, pagando o projeto e gerando lucro operacional.
* **Apoio a Stakeholders:** * **RH:** Direcionamento de verba para campanhas de conscientização focadas.
    * **Gestores:** Ferramenta preditiva para avaliação de segurança psicológica do time.
    * **Compliance:** Demonstração prática do compromisso corporativo com o *Duty of Care*.

## 🔬 Metodologia Técnica e Pré-processamento (EDA)

A análise exploratória profunda (presente no notebook `mental_tech_eda.ipynb`) guiou rigorosos processos de limpeza e engenharia de dados:

1. **Tratamento de Inconsistências:** Agrupamento de dezenas de inputs textuais livres na variável `Gender` (ex: "Male", "Cis Male", "m") em três categorias base (`Male`, `Female`, `Other`). Tratamento de outliers na variável idade.
2. **Imputação Inteligente:** Tratamento de valores nulos, com destaque para a variável crítica `work_interfere`, garantindo a integridade do dataset sem introduzir viés estatístico.
3. **Seleção Baseada em V de Cramér:** Variáveis categóricas foram selecionadas baseadas em sua real força de associação estatística com o *target*. Variáveis como `remote_work` e `tech_company` foram descartadas como ruído (score próximo a 0).

## 📊 Diagnóstico Estratégico (Insights de Negócio)

* **O Limbo das Médias Empresas:** Há uma falha crítica de comunicação em empresas de médio porte. O grupo que "Não sabe" se tem benefícios busca tratamento na mesma taxa de quem "Não tem". A ignorância sobre benefícios anula a existência deles.
* **Segurança Psicológica:** Existe uma relação inversa perfeita: o medo de consequências negativas na carreira zera a comunicação sobre saúde mental com supervisores diretos.
* **Estigma Demográfico:** O público masculino busca proporcionalmente menos ajuda do que o feminino, indicando a necessidade de campanhas com linguagem focada nesse demográfico resistente.

## 📈 Benchmarking e Performance (Curva ROC/AUC)

O **Gradient Boosting Classifier** foi selecionado após validação cruzada contra Regressão Logística e Random Forest:

* O modelo vencedor apresentou a melhor capacidade de distinção entre classes (AUC), com subida acentuada na Sensibilidade (Eixo Y), característica mandatória para modelos de triagem de saúde.
* O Random Forest apresentou menor capacidade de generalização para este conjunto específico de dados.

### Métricas Finais (Threshold Ajustado para 0.40)

| Métrica | Resultado | Interpretação de Negócio |
| :--- | :--- | :--- |
| **Recall (Sensibilidade)** | **87%** | **Foco Primário:** Identifica 87% dos colaboradores que realmente necessitam de ajuda. |
| **Precisão** | **69%** | Eficiência na gestão de Falsos Positivos (alarmes falsos aceitáveis em saúde preventiva). |
| **Falsos Negativos** | **17** | Risco minimizado: apenas 17 casos reais perdidos no conjunto de teste. |

## 💡 Feature Importance (O "Cérebro" do Modelo)

A árvore de decisão confirmou a hierarquia preditiva:

1. **Fatores Pessoais (Dominantes):** Histórico Familiar (`family_history`) e Interferência no Trabalho (`work_interfere`).
2. **Políticas Internas (Estratégicos):** Facilidade de solicitar licença e clareza nas opções de cuidado oferecidas.
3. **Demografia (Ruído):** Gênero e Nacionalidade não ditam a necessidade de tratamento de forma isolada, indicando que o adoecimento mental no setor tech é estrutural.

## 🛠️ Arquitetura do Repositório

```text
├── data/               # Raw survey data e CSV limpo (processed)
├── models/             # Artefatos serializados (.pkl) do Pipeline Final
├── notebooks/          # mental_tech_eda.ipynb (Limpeza, EDA e Testes)
├── src/                # Código de Produção (Scripts modulares)
├── main.py             # Orquestrador de Treinamento
├── app.py              # Interface Web interativa (Streamlit)
└── requirements.txt    # Dependências de ambiente (Python 3.11)
````

## 🚀 Como Executar o Projeto Localmente

```bash
# 1. Clone o repositório
git clone [https://github.com/felipetamiozzo/mental-health-predictor-ml.git](https://github.com/felipetamiozzo/mental-health-predictor-ml.git)
cd mental-health-predictor-ml

# 2. Instale as dependências
pip install -r requirements.txt

# 3. (Opcional) Retreine o modelo executando o pipeline
python main.py

# 4. Inicie o Web App
streamlit run app.py
```

-----

**Desenvolvido por [Felipe Tamiozzo](https://www.google.com/search?q=https://www.linkedin.com/in/felipe-tamiozzo) ** *Cientista de Dados focado em traduzir padrões matemáticos em estratégia de negócios.*

```
