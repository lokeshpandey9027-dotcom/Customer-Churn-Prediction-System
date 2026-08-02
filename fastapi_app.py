from fastapi import FastAPI, Request, Form, Cookie, Response, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import pickle
import os
import random
import io
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timedelta, timezone

# Load environment variables from .env file
load_dotenv()

# Securely fetch username and password strictly from environment variables
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")

# Setup SQLite Database for Customer Behavior & Churn Tracking
SQLALCHEMY_DATABASE_URL = "sqlite:///./customer_tracking.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

IST = timezone(timedelta(hours=5, minutes=30))

class CustomerActivityLog(Base):
    __tablename__ = "customer_activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, default=lambda: f"CUST-{random.randint(1000, 9999)}")
    timestamp = Column(DateTime, default=lambda: datetime.now(IST).replace(tzinfo=None))
    gender = Column(String)
    tenure = Column(Integer)
    contract = Column(String)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    prediction = Column(String)
    probability = Column(Float)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String)
    password = Column(String)

Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load model, encoders, and scaler
with open('best_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('encoder.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler_data = pickle.load(scaler_file)

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

def get_retention_suggestions(contract, tenure, online_security, tech_support, monthly_charges):
    """ Generates smart retention suggestions based on customer behavior/parameters """
    suggestions = []
    if contract == "Month-to-month":
        suggestions.append("💡 **Contract Upgrade Offer:** Offer a 10% discount to transition this customer to a 1-year or 2-year contract to secure long-term loyalty.")
    if online_security == "No" or tech_support == "No":
        suggestions.append("🛡️ **Security Bundle Pitch:** Provide a free 3-month trial of 'Online Security & Tech Support' add-ons to increase service stickiness.")
    if tenure < 12:
        suggestions.append("👋 **Onboarding Check-in:** Since tenure is low (< 1 year), trigger an automated customer satisfaction survey and dedicated support outreach.")
    if monthly_charges > 80:
        suggestions.append("🎁 **Loyalty Discount / Perk:** High monthly spender at risk. Proactively offer a loyalty bundle or reward points to offset bill fatigue.")
    
    if not suggestions:
        suggestions.append("✅ **Healthy Engagement:** Customer profile looks stable. Maintain standard automated newsletter and check-in rewards.")
    return suggestions

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])

    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])

    prediction = loaded_model.predict(input_df)[0]
    probability = float(loaded_model.predict_proba(input_df)[0, 1])
    res_label = "Churn" if prediction == 1 else "No Churn"
    return res_label, probability

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None, "success": None})

@app.post("/login", response_class=HTMLResponse)
async def login_post(response: Response, request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # Check default env user or registered database user
    valid_user = (username == USERNAME and password == PASSWORD)
    if not valid_user:
        db_user = db.query(User).filter(User.username == username, User.password == password).first()
        if db_user:
            valid_user = True

    if valid_user:
        redirect_response = RedirectResponse(url="/", status_code=303)
        redirect_response.set_cookie(key="session", value="authenticated", httponly=True)
        return redirect_response
    else:
        return templates.TemplateResponse(
            request=request, 
            name="login.html", 
            context={"error": "Invalid username or password", "success": None}
        )

@app.post("/register", response_class=HTMLResponse)
async def register_post(
    request: Request, 
    reg_username: str = Form(...), 
    reg_email: str = Form(...), 
    reg_password: str = Form(...), 
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(User.username == reg_username).first()
    if existing_user:
        return templates.TemplateResponse(
            request=request, 
            name="login.html", 
            context={"error": "Username already exists! Please choose another.", "success": None}
        )
    
    new_user = User(username=reg_username, email=reg_email, password=reg_password)
    db.add(new_user)
    db.commit()

    return templates.TemplateResponse(
        request=request, 
        name="login.html", 
        context={"error": None, "success": "Account created successfully! Please sign in using your credentials."}
    )

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="session")
    return response

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request, session: str = Cookie(None), db: Session = Depends(get_db)):
    if session != "authenticated":
        return RedirectResponse(url="/login", status_code=303)
    
    logs = db.query(CustomerActivityLog).order_by(CustomerActivityLog.timestamp.desc()).all()
    
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"prediction": None, "probability": None, "suggestions": None, "logs": logs, "success": None}
    )

@app.post("/", response_class=HTMLResponse)
async def process_prediction(
    request: Request,
    session: str = Cookie(None),
    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
    db: Session = Depends(get_db)
):
    if session != "authenticated":
        return RedirectResponse(url="/login", status_code=303)

    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
    }

    prediction, probability = make_prediction(input_data)
    suggestions = get_retention_suggestions(Contract, tenure, OnlineSecurity, TechSupport, MonthlyCharges)

    db_log = CustomerActivityLog(
        gender=gender,
        tenure=tenure,
        contract=Contract,
        monthly_charges=MonthlyCharges,
        total_charges=TotalCharges,
        prediction=prediction,
        probability=probability
    )
    db.add(db_log)
    db.commit()

    logs = db.query(CustomerActivityLog).order_by(CustomerActivityLog.timestamp.desc()).all()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": prediction, 
            "probability": probability,
            "suggestions": suggestions,
            "logs": logs,
            "success": None
        }
    )

@app.post("/bulk-upload", response_class=HTMLResponse)
async def bulk_upload(
    request: Request, 
    file: UploadFile = File(...), 
    session: str = Cookie(None), 
    db: Session = Depends(get_db)
):
    if session != "authenticated":
        return RedirectResponse(url="/login", status_code=303)
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Handle TotalCharges missing or space values
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).replace({" ": "0.0"}), errors='coerce').fillna(0.0)

        processed_count = 0
        for index, row in df.iterrows():
            # Build input dictionary required for prediction
            input_data = {
                'gender': str(row.get('gender', 'Male')),
                'SeniorCitizen': int(row.get('SeniorCitizen', 0)),
                'Partner': str(row.get('Partner', 'No')),
                'Dependents': str(row.get('Dependents', 'No')),
                'tenure': int(row.get('tenure', 1)),
                'PhoneService': str(row.get('PhoneService', 'Yes')),
                'MultipleLines': str(row.get('MultipleLines', 'No')),
                'InternetService': str(row.get('InternetService', 'DSL')),
                'OnlineSecurity': str(row.get('OnlineSecurity', 'No')),
                'OnlineBackup': str(row.get('OnlineBackup', 'No')),
                'DeviceProtection': str(row.get('DeviceProtection', 'No')),
                'TechSupport': str(row.get('TechSupport', 'No')),
                'StreamingTV': str(row.get('StreamingTV', 'No')),
                'StreamingMovies': str(row.get('StreamingMovies', 'No')),
                'Contract': str(row.get('Contract', 'Month-to-month')),
                'PaperlessBilling': str(row.get('PaperlessBilling', 'Yes')),
                'PaymentMethod': str(row.get('PaymentMethod', 'Electronic check')),
                'MonthlyCharges': float(row.get('MonthlyCharges', 20.0)),
                'TotalCharges': float(row.get('TotalCharges', 20.0))
            }

            prediction, probability = make_prediction(input_data)

            db_log = CustomerActivityLog(
                gender=input_data['gender'],
                tenure=input_data['tenure'],
                contract=input_data['Contract'],
                monthly_charges=input_data['MonthlyCharges'],
                total_charges=input_data['TotalCharges'],
                prediction=prediction,
                probability=probability
            )
            db.add(db_log)
            processed_count += 1
            
        db.commit()
        success_msg = f"Successfully processed and logged {processed_count} customer records from CSV file!"
    except Exception as e:
        success_msg = f"Error processing bulk file: {str(e)}"

    logs = db.query(CustomerActivityLog).order_by(CustomerActivityLog.timestamp.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": None, "probability": None, "suggestions": None, "logs": logs, "success": success_msg}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=True)