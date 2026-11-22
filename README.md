```markdown
# 🚀 Telco Customer Churn: Previsão com Rede Neural MLP e Deploy de API

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Jupyter](https://img.shields.io/badge/Jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## 📌 Sobre o Projeto
Este projeto foi desenvolvido como atividade avaliativa prática (AT2) com o objetivo de consolidar o ciclo de vida completo de um modelo de Aprendizagem de Máquina.

O problema de negócio escolhido foi a **previsão de Churn (cancelamento de clientes)** de uma empresa de telecomunicações, utilizando o dataset público [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) do Kaggle. A solução abrange desde a análise exploratória e treinamento de uma Rede Neural Multicamadas (MLP) até a disponibilização do modelo preditivo em produção através de uma API RESTful.

## 🛠️ Tecnologias e Ferramentas
- **Linguagem:** Python 3.10+
- **Gerenciador de Pacotes:** [UV](https://github.com/astral-sh/uv)
- **Manipulação de Dados e EDA:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `MLPClassifier`)
- **API e Servidor:** FastAPI, Uvicorn, Pydantic
- **Serialização de Modelo:** Joblib

## 📁 Estrutura do Repositório
A organização do projeto segue o padrão de microsserviços e separação de responsabilidades:

```text
telco-churn-mlp/
│
├── notebooks/
│   └── analise_treinamento.ipynb   # Notebook com EDA, treino e avaliação do modelo
│
├── models/
│   └── modelo_mlp.joblib           # Pipeline treinado e serializado (Pré-processamento + MLP)
│
├── src/
│   ├── __init__.py                 # Inicializador do módulo Python
│   ├── schemas.py                  # Contratos de dados (Validação com Pydantic)
│   └── app.py                      # Servidor FastAPI e rotas de inferência
│
├── requisicoes.http                # Casos de teste HTTP (Sucesso e Erro 422)
├── pyproject.toml                  # Configuração de dependências do UV
└── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset original
```

## ⚙️ Como Executar o Projeto Localmente

### 1. Pré-requisitos
Certifique-se de ter o Python e o [UV instalados](https://docs.astral.sh/uv/getting-started/installation/) em sua máquina.

### 2. Clonar o repositório e instalar dependências
```bash
git clone https://github.com/Leandro-dsm/telco-churn-mlp.git
cd telco-churn-mlp
uv sync
```

### 3. Rodar a API
O modelo já está treinado e serializado na pasta `models/`. Para subir o servidor de inferência, execute:
```bash
uv run fastapi dev src/app.py
```

A API estará disponível em `http://127.0.0.1:8000`.

## 🧠 Arquitetura da Solução
1. **Pré-processamento (Pipeline):** O `ColumnTransformer` aplica `StandardScaler` nas variáveis numéricas e `OneHotEncoder` nas categóricas. O pipeline completo é salvo via `joblib`, garantindo que a API receba os dados brutos e faça a transformação automaticamente, sem inconsistências.
2. **Rede Neural (MLP):** Utilizou-se o `MLPClassifier` com 2 camadas ocultas `(64, 32)`, taxa de aprendizado `0.001` e `early_stopping=True` para evitar overfitting.
3. **Validação (Pydantic):** O arquivo `schemas.py` garante que a API só aceite dados íntegros. Valores inválidos são bloqueados antes de chegarem ao modelo, retornando o status HTTP `422 Unprocessable Entity`.

## 📡 Endpoints da API

### `GET /info`
Retorna os metadados e o status atual do serviço.
**Exemplo de Retorno:**
```json
{
  "status": "ok",
  "modelo": "MLPClassifier (64, 32)",
  "dataset": "Telco Customer Churn",
  "target": "Churn (Yes/No)",
  "versao": "1.0"
}
```

### `POST /predict`
Recebe os dados de um cliente e retorna a predição e a probabilidade de Churn.
- A documentação interativa (Swagger UI) pode ser acessada em `http://127.0.0.1:8000/docs`.

**Exemplo de Request Body:**
```json
{
  "SeniorCitizen": 0,
  "tenure": 8,
  "MonthlyCharges": 99.65,
  "TotalCharges": 820.5,
  "gender": "Female",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```

**Exemplo de Retorno (HTTP 200):**
```json
{
  "classe_predita": "Yes",
  "probabilidade": 0.8542
}
```

## 🧪 Testes de Requisição
O arquivo `requisicoes.http` na raiz do projeto contém todos os cenários de teste (requisições válidas e inválidas). Para executá-los, utilize a extensão **REST Client** (VS Code) ou **Thunder Client**.

---
Desenvolvido por [Leandro S. Sueoka](https://github.com/Leandro-dsm) como parte da avaliação da disciplina de Aprendizagem de Máquina (FATEC).
```

---

### 💡 Por que este README é profissional?
1. **Badges visuais:** As etiquetas coloridas no topo dão um ar moderno e mostram rapidamente as tecnologias usadas.
2. **Contexto de negócio:** Ele não apenas diz "fiz um modelo", mas explica *por que* o Churn é importante.
3. **Estrutura clara:** Separação entre "O que é", "Como rodar", "Arquitetura" e "Endpoints", facilitando a leitura de qualquer pessoa (ou do seu professor na hora de avaliar).
4. **Exemplos reais:** Os JSONs de request e response mostram exatamente como interagir com a sua API.

**Próximo passo:** Depois de salvar o arquivo, rode os comandos no terminal para subir essa atualização:
```powershell
git add README.md
git commit -m "docs: adiciona README profissional"
git push
```
