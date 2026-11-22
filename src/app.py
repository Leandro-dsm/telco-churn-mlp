from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
import pandas as pd
from src.schemas import ClienteInput, PredicaoOutput


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o pipeline UMA VEZ no startup
    app.state.modelo = joblib.load("models/modelo_mlp.joblib")
    print("Modelo carregado no startup.")
    yield
    # shutdown: nada a fazer


app = FastAPI(
    title="API MLP — Telco Customer Churn",
    description="Predição de churn via MLPClassifier + Pipeline Scikit-Learn",
    lifespan=lifespan,
)


@app.get("/info")
def info():
    """Metadados e diagnóstico do serviço."""
    return {
        "status": "ok",
        "modelo": "MLPClassifier (64, 32)",
        "dataset": "Telco Customer Churn",
        "target": "Churn (Yes/No)",
        "versao": "1.0",
    }


@app.post("/predict", response_model=PredicaoOutput)
def predict(payload: ClienteInput):
    # 1. Converte Pydantic → DataFrame (mesmos nomes de coluna do treino)
    df = pd.DataFrame([payload.model_dump()])

    modelo = app.state.modelo

    # 2. Predição
    proba = modelo.predict_proba(df)[0]   # [proba_classe_0, proba_classe_1]
    classe = int(modelo.predict(df)[0])   # 0 ou 1

    # 3. Monta resposta
    return PredicaoOutput(
        classe_predita="Yes" if classe == 1 else "No",
        probabilidade=float(proba[classe]),
    )

