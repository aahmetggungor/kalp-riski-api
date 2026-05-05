from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# FastAPI uygulamasını başlatıyoruz
app = FastAPI(title="Kalp Hastaliği Risk Tahmin API")

# WEB (CHROME) ERİŞİMİ İÇİN CORS İZİNLERİ (Sihirli Kısım Burası)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Herkese (senin GitHub sitene de) izin ver
    allow_credentials=False, # Çakışmayı önlemek için False
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eğittiğimiz modeli ve ölçeklendiriciyi (scaler) hafızaya alıyoruz
model = joblib.load("kalp_riski_modeli.pkl")
scaler = joblib.load("scaler.pkl")

# Flutter'dan (Kullanıcıdan) gelecek JSON verisinin yapısını tanımlıyoruz
class PatientData(BaseModel):
    age: int
    ap_hi: int
    ap_lo: int
    cholesterol: int

# Tahmin yapacak POST endpoint'i
@app.post("/predict")
def predict_risk(data: PatientData):
    # Gelen veriyi modelin anlayacağı DataFrame formatına çeviriyoruz
    input_data = pd.DataFrame([{
        "age": data.age,
        "ap_hi": data.ap_hi,
        "ap_lo": data.ap_lo,
        "cholesterol": data.cholesterol
    }])
    
    # Modelin eğitimindeki gibi veriyi ölçeklendiriyoruz
    scaled_data = scaler.transform(input_data)
    
    # Gradient Boosting modelimize tahmin yaptırıyoruz
    prediction = model.predict(scaled_data)
    
    # Sonucu Flutter'ın kolayca okuyabileceği JSON formatında döndürüyoruz
    if prediction[0] == 1:
        return {"risk_status": 1, "message": "Risk Var"}
    else:
        return {"risk_status": 0, "message": "Risk Yok"}