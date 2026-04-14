# HOT RELOAD TRIGGER
import io, base64, warnings
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib, pickle
from collections import Counter

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import sqlite3
import json
import datetime
import os
import hashlib
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Use a dummy key to prevent immediate Uvicorn startup crash if Render Env Variables aren't set yet
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "dummy_key"))

DB_NAME = "neuroscan.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, user_id INTEGER, type TEXT, prediction TEXT, confidence REAL, date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS doctors (id INTEGER PRIMARY KEY, name TEXT, specialty TEXT, success_rate REAL, price INTEGER)''')
    
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        # Seed Mock Users
        mocks = [
            ("alice", "pass123"),
            ("marcus", "pass123"),
            ("chloe", "pass123"),
            ("jonathan", "pass123"),
            ("sofia", "pass123")
        ]
        for u, p in mocks:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, p))
            
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Seed Past Predictions
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (1, 'Biomarker', 'Mood Disorder', 89.2, ?)", (now,))
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (2, 'EEG CSV', 'Mood Disorder', 95.1, ?)", (now,))
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (3, 'Biomarker', 'Healthy', 98.7, ?)", (now,))
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (4, 'EEG CSV', 'Healthy', 82.4, ?)", (now,))
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (5, 'Biomarker', 'Mood Disorder', 76.5, ?)", (now,))
        
        # Seed Mock Doctors
        docs = [
            ("Dr. Evelyn Vance", "Cognitive Behavioral Therapy", 94.2, 120),
            ("Dr. Arthur Sterling", "Neuro-Linguistic Programming", 88.5, 95),
            ("Dr. Lisa Monroe", "Psychodynamic Regression", 91.0, 150),
            ("Dr. Marcus Chen", "Holistic EEG Mediation", 98.1, 200),
            ("Dr. Samuel Vane", "Somatic Experiencing", 85.3, 80)
        ]
        for name, spec, sr, price in docs:
            c.execute("INSERT INTO doctors (name, specialty, success_rate, price) VALUES (?, ?, ?, ?)", (name, spec, sr, price))

    conn.commit()
    conn.close()

init_db()

warnings.filterwarnings("ignore")

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request
import filetype
import logging

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="NeuroScan EEG API", description="Psychiatric disorder prediction from EEG signals.", version="2.0.0")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SAVE = "./"

try:
    full_pl       = joblib.load(SAVE + "eeg_xgboost_pipeline.pkl")
    FULL_MODEL    = full_pl["model"]
    FULL_SCALER   = full_pl["scaler"]
    FULL_SELECTOR = full_pl["selector"]
    FULL_FEATURES = full_pl["feature_names"]
    CLASSES       = full_pl["classes"]
    THRESHOLD     = full_pl.get("threshold", 0.50)
    print("[STARTUP] XGBoost pipeline loaded OK")
except Exception as e:
    print(f"[STARTUP ERROR] XGBoost: {e}")
    FULL_MODEL = None

try:
    bio_pl    = joblib.load(SAVE + "eeg_biomarker_manual_pipeline.pkl")
    BIO_MODEL  = bio_pl["model"]
    BIO_SCALER = bio_pl["scaler"]
    BIO_FEATS  = bio_pl["features"]
    BIO_RANGES = bio_pl["ranges"]
    print("[STARTUP] Biomarker pipeline loaded OK")
except Exception as e:
    print(f"[STARTUP ERROR] Biomarker: {e}")
    BIO_MODEL = None

class BiomarkerInput(BaseModel):
    faa:         float = Field(..., description="Frontal Alpha Asymmetry")
    theta_alpha: float = Field(..., description="Theta/Alpha ratio")
    beta_alpha:  float = Field(..., description="Beta/Alpha ratio")
    delta_asym:  float = Field(..., description="Delta Asymmetry")
    alpha_power: float = Field(..., description="Mean Alpha Power")
    theta_power: float = Field(..., description="Mean Theta Power")

class LoginInput(BaseModel):
    username: str
    password: str

class ChatInput(BaseModel):
    message: str
    context: Optional[str] = "Healthy"

class PredictionResponse(BaseModel):
    prediction:  str
    confidence:  float
    probability: dict
    explanation: str

class BookDoctorInput(BaseModel):
    username: str
    doctor_id: int

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def engineer_eeg_features(df: pd.DataFrame) -> pd.DataFrame:
    X_raw = df.copy()
    a_fp1 = X_raw.filter(like="alpha").filter(like="FP1")
    a_fp2 = X_raw.filter(like="alpha").filter(like="FP2")
    if not a_fp1.empty and not a_fp2.empty:
        df["FAA"] = a_fp1.values.mean(axis=1) - a_fp2.values.mean(axis=1)
    alpha_m = X_raw.filter(like="alpha").values.mean(axis=1)
    theta_m = X_raw.filter(like="theta").values.mean(axis=1)
    beta_m  = X_raw.filter(like="beta").values.mean(axis=1)
    df["theta_alpha_ratio"] = theta_m / (alpha_m + 1e-8)
    df["beta_alpha_ratio"]  = beta_m  / (alpha_m + 1e-8)
    d_fp1 = X_raw.filter(like="delta").filter(like="FP1")
    d_fp2 = X_raw.filter(like="delta").filter(like="FP2")
    if not d_fp1.empty and not d_fp2.empty:
        df["delta_asym"] = d_fp1.values.mean(axis=1) - d_fp2.values.mean(axis=1)
    df["alpha_power"] = alpha_m
    df["theta_power"] = theta_m
    return df

def build_explanation(faa, theta_alpha, beta_alpha, prob_mood) -> str:
    lines = []
    if faa < 0:
        lines.append("Negative FAA -> associated with depression tendency")
    if theta_alpha > 1.5:
        lines.append("Elevated Theta/Alpha ratio -> indicates cognitive slowing, fatigue")
    if beta_alpha > 1.5:
        lines.append("Elevated Beta/Alpha ratio -> indicates anxiety and hyperarousal")
    if prob_mood > 0.7:
        lines.append("High model confidence -> strong EEG signature match for mood disorder")
    if not lines:
        lines.append("EEG biomarkers within typical healthy range")
    return "\n".join(lines)

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "full_model": FULL_MODEL is not None, "bio_model": BIO_MODEL is not None, "classes": CLASSES, "version": "2.0.0"})

@app.get("/classes")
async def get_classes():
    return JSONResponse({"classes": CLASSES})

@app.get("/biomarker-ranges")
async def get_ranges():
    return JSONResponse({"ranges": BIO_RANGES, "features": BIO_FEATS})

def hash_pass(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

@app.post("/api/register")
def register(data: LoginInput):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (data.username, hash_pass(data.password)))
        conn.commit()
        return {"status": "success"}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username already exists")
    finally:
        conn.close()

@app.post("/api/login")
def login(data: LoginInput):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username=?", (data.username,))
    row = c.fetchone()
    conn.close()
    
    mock_users = ["alice", "marcus", "chloe", "jonathan", "sofia"]
    if data.password == "pass123" and data.username in mock_users:
        idx = mock_users.index(data.username) + 1
        return {"status": "success", "user_id": idx, "username": data.username}
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (data.username, hash_pass(data.password)))
    user = c.fetchone()
    conn.close()
    
    if user:
        return {"status": "success", "user_id": user[0], "username": data.username}
    raise HTTPException(401, "Invalid credentials")

@app.get("/api/stats")
def get_stats():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT u.username, p.type, p.prediction, p.confidence, p.date FROM predictions p JOIN users u ON p.user_id = u.id")
    rows = c.fetchall()
    conn.close()
    history = [{"username": r[0], "type": r[1], "prediction": r[2], "confidence": r[3], "date": r[4]} for r in rows]
    return {"history": history}

@app.post("/api/chat")
def chat(data: ChatInput):
    sys_prompt = f"You are Dr. NeuroScan, an AI Psychiatrist. The patient's last diagnosis is: {data.context}. Give clinical but highly approachable advice on medications, therapies, and lifestyle for this condition. Keep it concise, under 150 words."
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": data.message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=300,
        )
        return {"response": chat_completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(500, f"Groq Error: {str(e)}")

@app.post("/predict-csv")
@limiter.limit("10/minute")
async def predict_csv(request: Request, file: UploadFile = File(...)):
    if FULL_MODEL is None:
        raise HTTPException(503, "Full model not loaded.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")
    
    contents = await file.read()
    
    # File Size limit: max 5MB
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(413, "File too large. Maximum 5MB.")
        
    # MIME Validation via filetype (Pure Python, cloud safe)
    kind = filetype.guess(contents[:1024])
    # filetype returns None for plain text/csv since it only tracks binary magic signatures
    if kind is not None and kind.mime not in ["text/plain", "text/csv", "application/csv"]:
        raise HTTPException(400, "Invalid file format detected.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
        df = df.replace({"M": 1, "F": 0, "Male": 1, "Female": 0})
        drop_cols = ["label", "main.disorder", "specific.disorder", "no.", "eeg.date", "education"] + [c for c in df.columns if "Unnamed" in c]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df = df.fillna(df.median(numeric_only=True))
        df = engineer_eeg_features(df)
        df_aligned = df.reindex(columns=FULL_FEATURES, fill_value=0)
        df_aligned = df_aligned.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_sc  = FULL_SCALER.transform(df_aligned)
        X_sel = FULL_SELECTOR.transform(X_sc)
        probs = FULL_MODEL.predict_proba(X_sel)
        preds = (probs[:, 1] >= THRESHOLD).astype(int)
        pred_labels = [CLASSES[p] for p in preds]
        confidences = (probs.max(axis=1) * 100).tolist()
        dist = Counter(pred_labels)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#4ADE80" if k == "Healthy" else "#F87171" for k in dist.keys()]
        bars = ax.bar(dist.keys(), dist.values(), color=colors)
        ax.bar_label(bars, padding=3)
        ax.set_title("Predicted Class Distribution", fontsize=13)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        chart_b64 = fig_to_b64(fig)
        return JSONResponse({"n_samples": len(preds), "predictions": pred_labels, "confidences": confidences, "distribution": dict(dist), "dist_chart": chart_b64, "model_used": "XGBoost (full EEG features)", "threshold": THRESHOLD})
    except Exception as e:
        # Secure Error Logging: Log internally, generic HTTP message to prevent data leakage
        logging.error(f"Prediction error: {e}")
        raise HTTPException(500, "Prediction failed. Please check your data format.")

@app.post("/predict-biomarker", response_model=PredictionResponse)
async def predict_biomarker(data: BiomarkerInput):
    if BIO_MODEL is None:
        raise HTTPException(503, "Biomarker model not loaded.")
    try:
        raw = np.array([[data.faa, data.theta_alpha, data.beta_alpha, data.delta_asym]])
        raw[0, 1] = np.log1p(abs(raw[0, 1]))
        raw[0, 2] = np.log1p(abs(raw[0, 2]))
        X_sc      = BIO_SCALER.transform(raw)
        prob_mood = BIO_MODEL.predict_proba(X_sc)[0][1]
        if prob_mood >= 0.5:
            label      = "Mood Disorder"
            confidence = prob_mood * 100
        else:
            label      = "Healthy"
            confidence = (1 - prob_mood) * 100
        explanation = build_explanation(data.faa, data.theta_alpha, data.beta_alpha, prob_mood)
        return PredictionResponse(prediction=label, confidence=round(confidence, 2), probability={"Healthy": round((1 - prob_mood) * 100, 2), "Mood Disorder": round(prob_mood * 100, 2)}, explanation=explanation)
    except Exception as e:
        raise HTTPException(500, f"Biomarker prediction failed: {str(e)}")

@app.post("/api/chat")
async def chat_api(data: ChatInput):
    sys_prompt = f"You are Dr. Neuro, a psychiatric AI. Provide clinical advice based on EEG biomarkers. The user's last diagnosis context is: {data.context}. Give concise, empathetic, bulleted recommendations."
    try:
        reply = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": data.message}],
            max_tokens=150, temperature=0.7
        )
        return {"response": reply.choices[0].message.content}
    except Exception as e:
        return {"response": f"Clinical Database Offline: {str(e)}"}

@app.get("/api/doctors")
def get_doctors():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM doctors")
    docs = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"doctors": docs}

@app.post("/api/book")
def book_doctor(data: BookDoctorInput):
    return {"message": f"Successfully booked consultation! A calendar invite has been sent."}
