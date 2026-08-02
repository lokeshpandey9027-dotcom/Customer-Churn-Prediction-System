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
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta, timezone

# Load environment variables from .env file
load_dotenv()

# Securely fetch credentials and MongoDB URI from environment variables
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")
MONGO_URI = os.getenv("MONGODB_URI")

# Setup MongoDB Atlas connection using Motor (Async)
client = AsyncIOMotorClient(MONGO_URI)
db = client.customer_churn_database  # Database name
users_collection = db.users
logs_collection = db.customer_activity_logs

IST = timezone(timedelta(hours=5, minutes=30))

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
async def login_post(response: Response, request: Request, username: str = Form(...), password: str = Form(...)):
    # Check default env user or registered user in MongoDB
    valid_user = (username == USERNAME and password == PASSWORD)
    if not valid_user:
        db_user = await users_collection.find_one({"username": username, "password": password})
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
    reg_password: str = Form(...)
):
    existing_user = await users_collection.find_one({"username": reg_username})
    if existing_user:
        return templates.TemplateResponse(
            request=request, 
            name="login.html", 
            context={"error": "Username already exists! Please choose another.", "success": None}
        )
    
    await users_collection.insert_one({
        "username": reg_username,
        "email": reg_email,
        "password": reg_password
    })

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
async def read_form(request: Request, session: str = Cookie(None)):
    if session != "authenticated":
        return RedirectResponse(url="/login", status_code=303)
    
    # Fetch recent logs from MongoDB Atlas sorted by timestamp descending
    logs_cursor = logs_collection.find().sort("timestamp", -1).limit(50)
    logs = await logs_cursor.to_list(length=50)
    
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
    TotalCharges: float = Form(...)
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

    # Insert activity log into MongoDB Atlas
    log_doc = {
        "customer_id": f"CUST-{random.randint(1000, 9999)}",
        "timestamp": datetime.now(IST).replace(tzinfo=None),
        "gender": gender,
        "tenure": tenure,
        "contract": Contract,
        "monthly_charges": MonthlyCharges,
        "total_charges": TotalCharges,
        "prediction": prediction,
        "probability": probability
    }
    await logs_collection.insert_one(log_doc)

    logs_cursor = logs_collection.find().sort("timestamp", -1).limit(50)
    logs = await logs_cursor.to_list(length=50)

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
    session: str = Cookie(None)
):
    if session != "authenticated":
        return RedirectResponse(url="/login", status_code=303)
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).replace({" ": "0.0"}), errors='coerce').fillna(0.0)

        batch_docs = []
        processed_count = 0
        for index, row in df.iterrows():
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

            batch_docs.append({
                "customer_id": f"CUST-{random.randint(1000, 9999)}",
                "timestamp": datetime.now(IST).replace(tzinfo=None),
                "gender": input_data['gender'],
                "tenure": input_data['tenure'],
                "contract": input_data['Contract'],
                "monthly_charges": input_data['MonthlyCharges'],
                "total_charges": input_data['TotalCharges'],
                "prediction": prediction,
                "probability": probability
            })
            processed_count += 1
            
        if batch_docs:
            await logs_collection.insert_many(batch_docs)

        success_msg = f"Successfully processed and logged {processed_count} customer records to MongoDB Atlas!"
    except Exception as e:
        success_msg = f"Error processing bulk file: {str(e)}"

    logs_cursor = logs_collection.find().sort("timestamp", -1).limit(50)
    logs = await logs_cursor.to_list(length=50)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": None, "probability": None, "suggestions": None, "logs": logs, "success": success_msg}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=True)