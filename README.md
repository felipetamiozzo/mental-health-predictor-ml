# 🧠 Mental Health Predictor in Tech

## 📌 Visão Geral
Este projeto é uma solução de Machine Learning desenvolvida para identificar a probabilidade de colaboradores do setor de tecnologia necessitarem de tratamento para saúde mental. 

O objetivo é fornecer uma ferramenta de **triagem preventiva** para departamentos de RH e Saúde Ocupacional, permitindo ações proativas antes que quadros de burnout ou transtornos mentais se agravem.


## 💼 Problema de Negócio
A saúde mental no setor de tecnologia é um desafio crescente. Muitas vezes, o estigma ou a falta de clareza sobre os sintomas impedem que profissionais busquem ajuda.

**O desafio:** Como podemos utilizar dados demográficos e comportamentais para prever, com alta sensibilidade, se um colaborador precisa de suporte profissional?

**A Solução:** Um modelo preditivo focado em **Recall (Sensibilidade)**, priorizando a detecção de casos positivos para minimizar o risco de não identificar alguém que precisa de ajuda (Falso Negativo).

## 📊 Principais Insights (EDA)
A Análise Exploratória de Dados revelou padrões importantes:
1.  **Histórico Familiar:** É o fator preditivo mais forte. Colaboradores com histórico familiar de doenças mentais têm uma probabilidade significativamente maior de necessitar de tratamento.
2.  **Interferência no Trabalho:** A percepção subjetiva de que a saúde mental "interfere no trabalho" (mesmo que raramente) é um alerta vermelho crítico.
3.  **Ambiente Corporativo:** A existência de benefícios claros e facilidade para tirar licença médica influenciam positivamente a busca por tratamento.

## 🛠️ Metodologia e Tecnologias
* **Linguagem:** Python
* **Bibliotecas:** Pandas, Numpy, Seaborn, Matplotlib, Scikit-learn, Joblib.
* **Pipeline:**
    * Tratamento de dados nulos (Imputação).
    * Limpeza de dados categóricos.
    * *Feature Engineering* e *One-Hot Encoding*.
* **Modelo Final:** **Gradient Boosting Classifier**.
* **Estratégia de Tuning:** Otimização de hiperparâmetros com foco em maximizar o Recall, ajustando o *Decision Threshold* para **0.40**.

## 📈 Performance do Modelo
O modelo foi calibrado para atuar como uma ferramenta de **Screening (Triagem)**:

| Métrica | Resultado | Interpretação |
| :--- | :--- | :--- |
| **Recall (Sensibilidade)** | **87%** | De cada 100 casos reais, o modelo detecta 87. |
| **Precision** | **69%** | Quando o modelo alerta, ele está certo ~70% das vezes. |
| **Acurácia** | **73%** | Performance geral do modelo. |

> *Nota: A escolha por um threshold de 0.40 gera mais Falsos Positivos (alarmes falsos), uma decisão estratégica para priorizar o cuidado humano.*

## 📂 Estrutura do Projeto
```bash
├── data/
│   ├── raw/           #Dataset original
│   └── processed/     #Dados tratados
├── notebooks/         #Jupyter Notebooks (EDA e Modelagem)
├── src/               #Código Fonte da Aplicação
│   ├── app.py         #Interface Web (Streamlit)
│   └── modelo...pkl   #Modelo treinado e serializado
├── requirements.txt   #Dependências do projeto
└── README.md          #Documentação