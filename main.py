from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# FastAPI uygulamasını başlatıyoruz
app = FastAPI(title="Kalp Hastaliği Dereceli Risk Tahmin API")

# WEB (CHROME) ERİŞİMİ İÇİN CORS İZİNLERİ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eğittiğimiz modeli ve ölçeklendiriciyi (scaler) yüklüyoruz
model = joblib.load("kalp_riski_modeli.pkl")
scaler = joblib.load("scaler.pkl")

class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120) 
    ap_hi: int = Field(..., ge=60, le=250) 
    ap_lo: int = Field(..., ge=40, le=150) 
    cholesterol: int = Field(..., ge=1, le=3) 

@app.post("/predict")
def predict_risk(data: PatientData):
    input_data = pd.DataFrame([{
        "age": data.age,
        "ap_hi": data.ap_hi,
        "ap_lo": data.ap_lo,
        "cholesterol": data.cholesterol
    }])
    
    scaled_data = scaler.transform(input_data)
    
    # Olasılıkları hesaplıyoruz (Probability)
    probabilities = model.predict_proba(scaled_data)
    risk_yuzdesi = probabilities[0][1] * 100
    
    # Risk Seviyelerini Belirliyoruz (3 Seviyeli Mantık)
    if risk_yuzdesi < 35:
        return {
            "risk_status": 0,
            "message": "Düşük Risk",
            "probability": round(risk_yuzdesi, 2),
            "color": "Green"
        }
    elif 35 <= risk_yuzdesi < 70:
        return {
            "risk_status": 1,
            "message": "Orta Risk",
            "probability": round(risk_yuzdesi, 2),
            "color": "Orange"
        }
    else:
        return {
            "risk_status": 2,
            "message": "Yüksek Risk",
            "probability": round(risk_yuzdesi, 2),
            "color": "Red"
        }