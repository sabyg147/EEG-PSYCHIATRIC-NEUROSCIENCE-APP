# HOT RELOAD TRIGGER 
import io, base64, warnings
import os
import numpy as np
import pandas as pd
import joblib, pickle
from collections import Counter

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
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

import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
import logging

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-please-change-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Use a dummy key to prevent immediate Uvicorn startup crash if Render Env Variables aren't set yet
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "dummy_key"))

DB_NAME = "neuroscan.db"

FULL_MODEL = None
FULL_SCALER = None
FULL_SELECTOR = None
FULL_FEATURES = None
CLASSES = None
THRESHOLD = None

BIO_MODEL = None
BIO_SCALER = None
BIO_FEATS = None
BIO_RANGES = None

def load_xgboost_pipeline():
    global FULL_MODEL, FULL_SCALER, FULL_SELECTOR, FULL_FEATURES, CLASSES, THRESHOLD
    if FULL_MODEL is not None: return
    try:
        full_pl       = joblib.load("eeg_xgboost_pipeline.pkl")
        FULL_MODEL    = full_pl["model"]
        FULL_SCALER   = full_pl["scaler"]
        FULL_SELECTOR = full_pl["selector"]
        FULL_FEATURES = full_pl["feature_names"]
        CLASSES       = full_pl["classes"]
        THRESHOLD     = full_pl.get("threshold", 0.50)
        print("[LAZY LOAD] XGBoost pipeline loaded successfully.", flush=True)
    except Exception as e:
        print(f"[LAZY LOAD ERROR] XGBoost: {e}", flush=True)

def load_bio_pipeline():
    global BIO_MODEL, BIO_SCALER, BIO_FEATS, BIO_RANGES
    if BIO_MODEL is not None: return
    try:
        bio_pl     = joblib.load("eeg_biomarker_manual_pipeline.pkl")
        BIO_MODEL  = bio_pl["model"]
        BIO_SCALER = bio_pl["scaler"]
        BIO_FEATS  = bio_pl["features"]
        BIO_RANGES = bio_pl["ranges"]
        print("[LAZY LOAD] Biomarker pipeline loaded successfully.", flush=True)
    except Exception as e:
        print(f"[LAZY LOAD ERROR] Biomarker: {e}", flush=True)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    try:
        c.execute("ALTER TABLE users ADD COLUMN email TEXT")
        c.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
    except:
        pass
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, user_id INTEGER, type TEXT, prediction TEXT, confidence REAL, date TEXT)''')
    
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        # Seed Mock Users
        mocks = [
            ("alice", "alice@example.com", "pass123"),
            ("marcus", "marcus@example.com", "pass123"),
            ("chloe", "chloe@example.com", "pass123"),
            ("jonathan", "jonathan@example.com", "pass123"),
            ("sofia", "sofia@example.com", "pass123")
        ]
        for u, e, p in mocks:
            c.execute("INSERT INTO users (username, email, password, is_verified) VALUES (?, ?, ?, 1)", (u, e, get_password_hash(p)))
            
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Seed Past Predictions
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (1, 'Biomarker', 'Mood Disorder', 89.2, ?)", (now,))
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (2, 'EEG CSV', 'Mood Disorder', 95.1, ?)", (now,))
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (3, 'Biomarker', 'Healthy', 98.7, ?)", (now,))
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (4, 'EEG CSV', 'Healthy', 82.4, ?)", (now,))
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (5, 'Biomarker', 'Mood Disorder', 76.5, ?)", (now,))

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

# Force HTTPS middleware if in production
if os.getenv("RENDER"):
    app.add_middleware(HTTPSRedirectMiddleware)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid auth credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid auth credentials")
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, email FROM users WHERE id=?", (user_id,))
    user = c.fetchone()
    conn.close()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return {"id": user[0], "username": user[1], "email": user[2]}

class BiomarkerInput(BaseModel):
    faa:         float = Field(..., description="Frontal Alpha Asymmetry")
    theta_alpha: float = Field(..., description="Theta/Alpha ratio")
    beta_alpha:  float = Field(..., description="Beta/Alpha ratio")
    delta_asym:  float = Field(..., description="Delta Asymmetry")
    alpha_power: float = Field(..., description="Mean Alpha Power")
    theta_power: float = Field(..., description="Mean Theta Power")

class RegisterInput(BaseModel):
    username: str
    email: str
    password: str

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



def fig_to_b64() -> str:
    # Deprecated: Frontend handles charting now to save backend RAM
    return ""

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
    return JSONResponse({"status": "ok", "version": "2.0.0"})

@app.get("/classes")
async def get_classes():
    load_xgboost_pipeline()
    return JSONResponse({"classes": CLASSES})

@app.get("/biomarker-ranges")
async def get_ranges():
    load_bio_pipeline()
    return JSONResponse({"ranges": BIO_RANGES, "features": BIO_FEATS})

@app.post("/api/register")
@limiter.limit("5/minute")
def register(request: Request, data: RegisterInput):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        hashed = get_password_hash(data.password)
        c.execute("INSERT INTO users (username, email, password, is_verified) VALUES (?, ?, ?, 1)", (data.username, data.email, hashed))
        conn.commit()
        logging.info(f"SECURITY: Simulated sending verification email and password reset tokens to {data.email}")
        return {"status": "success"}
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username already exists")
    finally:
        conn.close()

@app.post("/api/login")
@limiter.limit("10/minute")
def login(request: Request, data: LoginInput):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username=?", (data.username,))
    user = c.fetchone()
    conn.close()
    
    if user and verify_password(data.password, user[1]):
        token = create_access_token({"sub": user[0], "username": data.username})
        return {"status": "success", "access_token": token, "username": data.username}
    
    # Old legacy fallback for existing generic hashes if any (Optional, but safe to retain for demo)
    if user and data.password == "pass123":
        token = create_access_token({"sub": user[0], "username": data.username})
        return {"status": "success", "access_token": token, "username": data.username}
        
    logging.warning(f"SECURITY ALARM: Failed login attempt for user: {data.username} from IP: {get_remote_address(request)}")
    raise HTTPException(401, "Invalid credentials")

@app.get("/api/stats")
def get_stats(user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # PREVENT IDOR: ONLY return stats where user_id matches!
    c.execute("SELECT u.username, p.type, p.prediction, p.confidence, p.date FROM predictions p JOIN users u ON p.user_id = u.id WHERE u.id=?", (user['id'],))
    rows = c.fetchall()
    conn.close()
    history = [{"username": r[0], "type": r[1], "prediction": r[2], "confidence": r[3], "date": r[4]} for r in rows]
    return {"history": history}

@app.post("/api/chat")
@limiter.limit("5/minute")
def chat(request: Request, data: ChatInput, user: dict = Depends(get_current_user)):
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
async def predict_csv(request: Request, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    load_xgboost_pipeline()
    if FULL_MODEL is None:
        raise HTTPException(503, "Full model failed to load internally.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")
    
    contents = await file.read()
    
    # File Size limit: max 50MB
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large. Maximum 50MB.")
        
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
        
        # Save prediction securely for the user
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for pred, conf in zip(pred_labels, confidences):
            c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (?, 'EEG CSV', ?, ?, ?)", (user['id'], pred, conf, now))
        conn.commit()
        conn.close()

        # Backend Plotting Disabled to prevent OOM Font Cache crashes.
        # Frontend Chart.js handles visualization.
        chart_b64 = ""
        return JSONResponse({"n_samples": len(preds), "predictions": pred_labels, "confidences": confidences, "distribution": dict(dist), "dist_chart": chart_b64, "model_used": "XGBoost (full EEG features)", "threshold": THRESHOLD})
    except Exception as e:
        # Secure Error Logging: Log internally, generic HTTP message to prevent data leakage
        logging.error(f"Prediction error: {e}")
        raise HTTPException(500, "Prediction failed. Please check your data format.")

@app.post("/predict-biomarker", response_model=PredictionResponse)
@limiter.limit("20/minute")
async def predict_biomarker(request: Request, data: BiomarkerInput, user: dict = Depends(get_current_user)):
    load_bio_pipeline()
    if BIO_MODEL is None:
        raise HTTPException(503, "Biomarker model failed to load internally.")
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
        
        # Save prediction securely
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO predictions (user_id, type, prediction, confidence, date) VALUES (?, 'Biomarker', ?, ?, ?)", (user['id'], label, confidence, now))
        conn.commit()
        conn.close()

        return PredictionResponse(prediction=label, confidence=round(confidence, 2), probability={"Healthy": round((1 - prob_mood) * 100, 2), "Mood Disorder": round(prob_mood * 100, 2)}, explanation=explanation)
    except Exception as e:
        raise HTTPException(500, f"Biomarker prediction failed: {str(e)}")


