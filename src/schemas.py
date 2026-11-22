from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    # --- Numéricas ---
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 se idoso, 0 caso contrário", examples=[0])
    tenure: int = Field(..., ge=0, le=100, description="Meses de relacionamento", examples=[24])
    MonthlyCharges: float = Field(..., ge=0, le=500, description="Cobrança mensal (R$)", examples=[70.35])
    TotalCharges: float = Field(..., ge=0, description="Total já pago (R$)", examples=[1680.0])

    # --- Categóricas ---
    gender: str = Field(..., examples=["Female"])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["Yes"])
    OnlineBackup: str = Field(..., examples=["No"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])


class PredicaoOutput(BaseModel):
    classe_predita: str = Field(..., description="'Yes' (vai cancelar) ou 'No' (permanece)")
    probabilidade: float = Field(..., ge=0.0, le=1.0, description="Confiança da predição")