"""
FastAPI app: home, dashboard, and prediction routes.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI(title="Energy Consumption Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

bundle = joblib.load("model.joblib")
scaler = bundle["scaler"]
pca = bundle["pca"]
model = bundle["model"]
FEATURES = bundle["feature_names"]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):
    return templates.TemplateResponse(
        "predict.html", {"request": request, "features": FEATURES, "prediction": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    try:
        values = [float(form.get(f, 0)) for f in FEATURES]
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        pred = float(model.predict(X_pca)[0])
    except Exception as e:
        pred = f"Error: {e}"

    return templates.TemplateResponse(
        "predict.html",
        {"request": request, "features": FEATURES, "prediction": pred},
    )
