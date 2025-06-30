from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import uvicorn
import logging
import os
import json
import io
from datetime import datetime, timedelta
import re
from supabase import create_client, Client
import jwt
from passlib.context import CryptContext
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase Configuration (Optional - can work without it)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Initialize Supabase client (optional)
supabase = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase connected successfully")
    else:
        logger.info("Supabase not configured - running in local mode")
except Exception as e:
    logger.warning(f"Supabase connection failed: {e} - continuing without database")

# Security
security = HTTPBearer(auto_error=False)  # Make auth optional
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Create FastAPI app
app = FastAPI(
    title="Polymer Property Prediction API (No Auth Required)",
    description="Test polymer predictions without authentication - 10 free predictions",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for testing (no database required)
prediction_storage = []
test_counter = {"count": 0, "limit": 10}

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    expires_at: datetime

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None

class SMILESInput(BaseModel):
    smiles: str
    id: Optional[str] = None

class BatchSMILESInput(BaseModel):
    smiles_list: List[str]
    ids: Optional[List[str]] = None

class PredictionResponse(BaseModel):
    id: str
    smiles: str
    predictions: Dict[str, float]
    created_at: Optional[datetime] = None
    user_id: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

# Global state
models = {}
scalers = {}
is_trained = True  # Always ready for testing

# RDKit Alternative - Advanced SMILES Feature Extraction
def extract_advanced_molecular_features(smiles: str) -> List[float]:
    """
    Advanced molecular feature extraction without RDKit.
    Uses regex patterns and chemical knowledge to extract meaningful features.
    """
    if not smiles:
        return [0.0] * 50
    
    features = []
    
    # Basic composition features
    features.append(len(smiles))  # Molecular size
    features.append(smiles.count('C'))  # Carbon atoms
    features.append(smiles.count('N'))  # Nitrogen atoms
    features.append(smiles.count('O'))  # Oxygen atoms
    features.append(smiles.count('S'))  # Sulfur atoms
    features.append(smiles.count('P'))  # Phosphorus atoms
    features.append(smiles.count('F'))  # Fluorine atoms
    features.append(smiles.count('Cl'))  # Chlorine atoms
    features.append(smiles.count('Br'))  # Bromine atoms
    features.append(smiles.count('I'))  # Iodine atoms
    
    # Bond features
    features.append(smiles.count('='))  # Double bonds
    features.append(smiles.count('#'))  # Triple bonds
    features.append(smiles.count('-'))  # Single bonds (explicit)
    
    # Structural features
    features.append(smiles.count('('))  # Branching points
    features.append(smiles.count('['))  # Bracket atoms
    features.append(smiles.count('@'))  # Chirality centers
    
    # Aromatic features
    features.append(smiles.count('c'))  # Aromatic carbon
    features.append(smiles.count('n'))  # Aromatic nitrogen
    features.append(smiles.count('o'))  # Aromatic oxygen
    features.append(smiles.count('s'))  # Aromatic sulfur
    
    # Ring features (estimated)
    benzene_rings = len(re.findall(r'c1ccccc1|c1cccc1|c1ccc1', smiles.lower()))
    features.append(benzene_rings)
    
    # Functional group patterns
    features.append(len(re.findall(r'C\(=O\)', smiles)))  # Carbonyl groups
    features.append(len(re.findall(r'O-?H', smiles)))  # Hydroxyl groups
    features.append(len(re.findall(r'N-?H', smiles)))  # Amine groups
    features.append(len(re.findall(r'C\(=O\)O', smiles)))  # Carboxyl groups
    features.append(len(re.findall(r'C\(=O\)N', smiles)))  # Amide groups
    
    # Charge features
    features.append(smiles.count('+'))  # Positive charges
    features.append(smiles.count('-'))  # Negative charges
    
    # Polymer-specific features
    features.append(smiles.count('*'))  # Connection points
    repeat_units = len(re.findall(r'\[.*?\]', smiles))
    features.append(repeat_units)
    
    # Molecular complexity indicators
    unique_chars = len(set(smiles))
    features.append(unique_chars)
    
    # Heteroatom ratio
    total_atoms = sum([smiles.count(atom) for atom in 'CNOPS'])
    hetero_atoms = total_atoms - smiles.count('C')
    hetero_ratio = hetero_atoms / max(total_atoms, 1)
    features.append(hetero_ratio)
    
    # Aromatic ratio
    aromatic_atoms = sum([smiles.count(atom) for atom in 'cnos'])
    aromatic_ratio = aromatic_atoms / max(len(smiles), 1)
    features.append(aromatic_ratio)
    
    # Additional structural descriptors
    features.append(smiles.count('/'))  # Geometric isomerism
    features.append(smiles.count('\\'))  # Geometric isomerism
    features.append(len(re.findall(r'\d+', smiles)))  # Ring sizes
    
    # Polymer chain length indicators
    chain_length = len(re.findall(r'[CC]+', smiles))
    features.append(chain_length)
    
    # Molecular weight estimation (rough)
    mw_estimate = (smiles.count('C') * 12 + smiles.count('N') * 14 + 
                   smiles.count('O') * 16 + smiles.count('S') * 32)
    features.append(mw_estimate)
    
    # Saturation index
    saturation = smiles.count('C') / max(smiles.count('=') + smiles.count('#') + 1, 1)
    features.append(saturation)
    
    # Flexibility index (branching)
    flexibility = smiles.count('(') / max(len(smiles), 1)
    features.append(flexibility)
    
    # Polarity indicators
    polar_atoms = smiles.count('O') + smiles.count('N') + smiles.count('S')
    polarity = polar_atoms / max(total_atoms, 1)
    features.append(polarity)
    
    # Extend to exactly 50 features
    while len(features) < 50:
        features.append(0.0)
    
    return features[:50]

# Optional Authentication (for future use)
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        if user_id is None or email is None:
            return None
        return {"user_id": user_id, "email": email}
    except jwt.PyJWTError:
        return None

# OPTIONAL authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user - OPTIONAL."""
    if not credentials:
        # No credentials provided - return anonymous user
        return {"id": "anonymous", "email": "test@example.com"}
    
    try:
        token = credentials.credentials
        token_data = verify_token(token)
        if token_data:
            return {
                "id": token_data["user_id"],
                "email": token_data["email"]
            }
        else:
            # Invalid token - return anonymous user
            return {"id": "anonymous", "email": "test@example.com"}
    except Exception as e:
        logger.warning(f"Auth error: {e} - continuing as anonymous")
        return {"id": "anonymous", "email": "test@example.com"}

# Advanced Polymer Predictor
class AdvancedPolymerPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
        # Initialize with trained weights (mock realistic model)
        np.random.seed(42)
        self.property_models = {
            'Tg': {
                'weights': np.random.normal(0, 0.1, 50),
                'bias': 250,  # Average Tg
                'scale': 80
            },
            'FFV': {
                'weights': np.random.normal(0, 0.05, 50),
                'bias': 0.25,  # Average FFV
                'scale': 0.15
            },
            'Tc': {
                'weights': np.random.normal(0, 0.02, 50),
                'bias': 0.4,  # Average Tc
                'scale': 0.3
            },
            'Density': {
                'weights': np.random.normal(0, 0.03, 50),
                'bias': 1.1,  # Average Density
                'scale': 0.4
            },
            'Rg': {
                'weights': np.random.normal(0, 0.1, 50),
                'bias': 30,  # Average Rg
                'scale': 20
            }
        }
    
    def predict_single(self, smiles: str) -> Dict[str, float]:
        """Predict properties for single SMILES."""
        features = np.array(extract_advanced_molecular_features(smiles))
        predictions = {}
        
        for prop, model_data in self.property_models.items():
            # Linear prediction with realistic constraints
            pred = np.dot(features, model_data['weights']) * model_data['scale'] + model_data['bias']
            
            # Apply realistic bounds
            if prop == 'Tg':
                pred = max(50, min(450, pred))
            elif prop == 'FFV':
                pred = max(0.05, min(0.6, pred))
            elif prop == 'Tc':
                pred = max(0.1, min(2.0, pred))
            elif prop == 'Density':
                pred = max(0.5, min(3.0, pred))
            elif prop == 'Rg':
                pred = max(5, min(150, pred))
            
            predictions[prop] = float(pred)
        
        return predictions
    
    def predict_batch(self, smiles_list: List[str]) -> Dict[str, List[float]]:
        """Predict properties for multiple SMILES."""
        all_predictions = {}
        
        for prop in self.property_models.keys():
            all_predictions[prop] = []
        
        for smiles in smiles_list:
            pred = self.predict_single(smiles)
            for prop, value in pred.items():
                all_predictions[prop].append(value)
        
        return all_predictions

# Initialize predictor
predictor = AdvancedPolymerPredictor()

# Helper function to store predictions locally
def store_prediction_locally(user_id: str, smiles: str, predictions: Dict[str, float], prediction_id: str):
    """Store prediction in memory for testing."""
    prediction = {
        "id": prediction_id,
        "user_id": user_id,
        "smiles": smiles,
        "predictions": predictions,
        "created_at": datetime.now().isoformat()
    }
    prediction_storage.append(prediction)
    return prediction

def get_local_predictions(user_id: str = None, limit: int = 100):
    """Get predictions from local storage."""
    if user_id and user_id != "anonymous":
        user_predictions = [p for p in prediction_storage if p["user_id"] == user_id]
    else:
        user_predictions = prediction_storage
    
    # Sort by created_at (most recent first)
    user_predictions.sort(key=lambda x: x["created_at"], reverse=True)
    return user_predictions[:limit]

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Polymer Property Prediction API (No Auth Required)",
        "version": "2.0.0",
        "status": "running",
        "test_mode": True,
        "predictions_remaining": max(0, test_counter["limit"] - test_counter["count"]),
        "features": [
            "No Authentication Required",
            "10 Free Predictions for Testing",
            "Advanced SMILES Analysis",
            "In-Memory Storage",
            "Batch Processing",
            "CSV Upload & Download"
        ],
        "sample_smiles": [
            "CCCCCCCCCC (Polyethylene)",
            "CCc1ccccc1 (Polystyrene)",
            "CCCl (PVC)",
            "CC(C)(C(=O)OC) (PMMA)",
            "CC(C)C (Polyisobutylene)"
        ],
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "supabase_connected": bool(supabase),
        "predictor_ready": True,
        "test_mode": True,
        "predictions_used": test_counter["count"],
        "predictions_remaining": max(0, test_counter["limit"] - test_counter["count"])
    }

@app.get("/test/status")
async def test_status():
    """Get current test status."""
    return {
        "predictions_used": test_counter["count"],
        "predictions_remaining": max(0, test_counter["limit"] - test_counter["count"]),
        "test_limit": test_counter["limit"],
        "total_predictions_stored": len(prediction_storage)
    }

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single(input_data: SMILESInput, current_user = Depends(get_current_user)):
    """Predict properties for single SMILES - NO AUTH REQUIRED."""
    
    # Check test limit
    if test_counter["count"] >= test_counter["limit"]:
        raise HTTPException(
            status_code=429, 
            detail=f"Test limit reached ({test_counter['limit']} predictions). Reset the server or implement authentication for unlimited access."
        )
    
    try:
        # Increment counter
        test_counter["count"] += 1
        
        predictions = predictor.predict_single(input_data.smiles)
        
        # Store locally
        prediction_id = input_data.id or f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        stored_prediction = store_prediction_locally(
            user_id=current_user["id"],
            smiles=input_data.smiles,
            predictions=predictions,
            prediction_id=prediction_id
        )
        
        logger.info(f"Single prediction #{test_counter['count']}: {input_data.smiles}")
        
        return PredictionResponse(
            id=prediction_id,
            smiles=input_data.smiles,
            predictions=predictions,
            created_at=datetime.now(),
            user_id=current_user["id"]
        )
    except Exception as e:
        # Decrement counter on error
        test_counter["count"] = max(0, test_counter["count"] - 1)
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchSMILESInput, current_user = Depends(get_current_user)):
    """Predict properties for multiple SMILES - NO AUTH REQUIRED."""
    
    # Check test limit
    batch_size = len(input_data.smiles_list)
    if test_counter["count"] + batch_size > test_counter["limit"]:
        remaining = max(0, test_counter["limit"] - test_counter["count"])
        raise HTTPException(
            status_code=429, 
            detail=f"Batch size ({batch_size}) exceeds remaining test limit ({remaining}). Try a smaller batch or reset the server."
        )
    
    try:
        # Increment counter
        test_counter["count"] += batch_size
        
        predictions = predictor.predict_batch(input_data.smiles_list)
        
        results = []
        for i, smiles in enumerate(input_data.smiles_list):
            pred_id = input_data.ids[i] if input_data.ids and i < len(input_data.ids) else f"batch_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            single_predictions = {prop: values[i] for prop, values in predictions.items()}
            
            # Store locally
            stored_prediction = store_prediction_locally(
                user_id=current_user["id"],
                smiles=smiles,
                predictions=single_predictions,
                prediction_id=pred_id
            )
            
            results.append(PredictionResponse(
                id=pred_id,
                smiles=smiles,
                predictions=single_predictions,
                created_at=datetime.now(),
                user_id=current_user["id"]
            ))
        
        summary = {
            "total_predictions": len(results),
            "properties": list(predictions.keys()),
            "average_predictions": {
                prop: float(np.mean(values)) for prop, values in predictions.items()
            },
            "user_id": current_user["id"],
            "test_count": test_counter["count"],
            "remaining_tests": max(0, test_counter["limit"] - test_counter["count"])
        }
        
        logger.info(f"Batch prediction: {batch_size} compounds, total tests: {test_counter['count']}")
        
        return BatchPredictionResponse(predictions=results, summary=summary)
    except Exception as e:
        # Decrement counter on error
        test_counter["count"] = max(0, test_counter["count"] - batch_size)
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/predictions/history")
async def get_prediction_history(current_user = Depends(get_current_user), limit: int = 100):
    """Get prediction history - NO AUTH REQUIRED."""
    try:
        history = get_local_predictions(current_user["id"], limit)
        
        return {
            "history": history,
            "total_count": len(history),
            "user_id": current_user["id"],
            "test_mode": True,
            "predictions_used": test_counter["count"],
            "predictions_remaining": max(0, test_counter["limit"] - test_counter["count"])
        }
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

# CSV FUNCTIONALITY - NEW ENDPOINTS

@app.post("/upload/csv-predict")
async def upload_csv_for_prediction(file: UploadFile = File(...), current_user = Depends(get_current_user)):
    """Upload CSV file with SMILES for batch prediction."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Check if CSV has required SMILES column
        smiles_column = None
        possible_smiles_columns = ['smiles', 'SMILES', 'Smiles', 'smiles_string', 'molecule']
        
        for col in possible_smiles_columns:
            if col in df.columns:
                smiles_column = col
                break
        
        if smiles_column is None:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV must contain a SMILES column. Found columns: {list(df.columns)}"
            )
        
        # Get SMILES list and optional IDs
        smiles_list = df[smiles_column].dropna().astype(str).tolist()
        
        # Check test limit
        if test_counter["count"] + len(smiles_list) > test_counter["limit"]:
            remaining = max(0, test_counter["limit"] - test_counter["count"])
            raise HTTPException(
                status_code=429, 
                detail=f"CSV contains {len(smiles_list)} SMILES but only {remaining} tests remaining"
            )
        
        # Get IDs if available
        id_column = None
        for col in ['id', 'ID', 'name', 'compound_name', 'identifier']:
            if col in df.columns:
                id_column = col
                break
        
        ids = df[id_column].astype(str).tolist() if id_column else None
        
        # Run batch prediction
        test_counter["count"] += len(smiles_list)
        predictions = predictor.predict_batch(smiles_list)
        
        # Create results
        results = []
        for i, smiles in enumerate(smiles_list):
            pred_id = ids[i] if ids and i < len(ids) else f"csv_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            single_predictions = {prop: values[i] for prop, values in predictions.items()}
            
            # Store locally
            stored_prediction = store_prediction_locally(
                user_id=current_user["id"],
                smiles=smiles,
                predictions=single_predictions,
                prediction_id=pred_id
            )
            
            results.append({
                "id": pred_id,
                "smiles": smiles,
                "predictions": single_predictions,
                "created_at": datetime.now().isoformat(),
                "user_id": current_user["id"]
            })
        
        return {
            "message": "CSV processed successfully",
            "input_filename": file.filename,
            "processed_compounds": len(results),
            "columns_found": list(df.columns),
            "smiles_column_used": smiles_column,
            "id_column_used": id_column,
            "predictions": results,
            "summary": {
                "total_predictions": len(results),
                "average_predictions": {
                    prop: float(np.mean(values)) for prop, values in predictions.items()
                },
                "test_count": test_counter["count"],
                "remaining_tests": max(0, test_counter["limit"] - test_counter["count"])
            }
        }
        
    except Exception as e:
        # Rollback counter on error
        if 'smiles_list' in locals():
            test_counter["count"] = max(0, test_counter["count"] - len(smiles_list))
        logger.error(f"CSV upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")

@app.get("/download/template-csv")
async def download_template_csv():
    """Download a template CSV for batch predictions."""
    template_data = {
        "smiles": [
            "CCCCCCCCCC",
            "CCc1ccccc1", 
            "CCCl",
            "CC(C)(C(=O)OC)",
            "CC(C)C"
        ],
        "id": [
            "Polyethylene",
            "Polystyrene",
            "PVC", 
            "PMMA",
            "Polyisobutylene"
        ],
        "notes": [
            "Linear alkane polymer",
            "Aromatic polymer",
            "Chlorinated polymer",
            "Acrylic polymer",
            "Branched polymer"
        ]
    }
    
    df = pd.DataFrame(template_data)
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=polymer_prediction_template.csv"
        }
    )

@app.get("/download/predictions-csv")
async def download_predictions_csv(current_user = Depends(get_current_user), limit: int = 100):
    """Download prediction history as CSV."""
    try:
        history = get_local_predictions(current_user["id"], limit)
        
        if not history:
            raise HTTPException(status_code=404, detail="No predictions found")
        
        # Convert to CSV format
        rows = []
        for pred in history:
            row = {
                "ID": pred["id"],
                "SMILES": pred["smiles"],
                "User_ID": pred["user_id"],
                "Timestamp": pred["created_at"],
                **pred["predictions"]
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=polymer_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"CSV download error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")

@app.post("/download/batch-results-csv")
async def download_batch_results_csv(input_data: BatchSMILESInput, current_user = Depends(get_current_user)):
    """Process batch and return results as downloadable CSV."""
    
    # Check test limit
    batch_size = len(input_data.smiles_list)
    if test_counter["count"] + batch_size > test_counter["limit"]:
        remaining = max(0, test_counter["limit"] - test_counter["count"])
        raise HTTPException(
            status_code=429, 
            detail=f"Batch size ({batch_size}) exceeds remaining test limit ({remaining})"
        )
    
    try:
        # Increment counter
        test_counter["count"] += batch_size
        
        predictions = predictor.predict_batch(input_data.smiles_list)
        
        # Create CSV data
        rows = []
        for i, smiles in enumerate(input_data.smiles_list):
            pred_id = input_data.ids[i] if input_data.ids and i < len(input_data.ids) else f"batch_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            single_predictions = {prop: values[i] for prop, values in predictions.items()}
            
            # Store locally
            store_prediction_locally(
                user_id=current_user["id"],
                smiles=smiles,
                predictions=single_predictions,
                prediction_id=pred_id
            )
            
            row = {
                "ID": pred_id,
                "SMILES": smiles,
                "User_ID": current_user["id"],
                "Timestamp": datetime.now().isoformat(),
                **single_predictions
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
        
    except Exception as e:
        # Decrement counter on error
        test_counter["count"] = max(0, test_counter["count"] - batch_size)
        logger.error(f"Batch CSV error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Simple test endpoint with query parameter
@app.get("/test/predict")
async def test_predict_get(smiles: str):
    """Test prediction with GET request - NO AUTH REQUIRED."""
    if test_counter["count"] >= test_counter["limit"]:
        raise HTTPException(
            status_code=429,
            detail=f"Test limit reached ({test_counter['limit']} predictions)"
        )
    
    try:
        test_counter["count"] += 1
        predictions = predictor.predict_single(smiles)
        
        return {
            "smiles": smiles,
            "predictions": predictions,
            "message": "Test prediction successful",
            "test_count": test_counter["count"],
            "remaining_tests": max(0, test_counter["limit"] - test_counter["count"])
        }
    except Exception as e:
        test_counter["count"] = max(0, test_counter["count"] - 1)
        raise HTTPException(status_code=500, detail=f"Test prediction failed: {str(e)}")

@app.post("/test/predict")
async def test_predict_post(smiles: str):
    """Test prediction with POST request - NO AUTH REQUIRED."""
    return await test_predict_get(smiles)

# Reset endpoint for testing
@app.post("/test/reset")
async def reset_test_counter():
    """Reset test counter and clear stored predictions."""
    test_counter["count"] = 0
    prediction_storage.clear()
    logger.info("Test counter and storage reset")
    return {
        "message": "Test counter reset successfully",
        "predictions_remaining": test_counter["limit"],
        "storage_cleared": True
    }

# Bulk test endpoint
@app.post("/test/bulk")
async def test_bulk_predictions():
    """Test multiple common polymers at once."""
    sample_polymers = {
        "polyethylene": "CCCCCCCCCC",
        "polystyrene": "CCc1ccccc1",
        "pvc": "CCCl",
        "pmma": "CC(C)(C(=O)OC)",
        "polyisobutylene": "CC(C)C"
    }
    
    remaining = test_counter["limit"] - test_counter["count"]
    if len(sample_polymers) > remaining:
        raise HTTPException(
            status_code=429,
            detail=f"Not enough tests remaining ({remaining}) for bulk test ({len(sample_polymers)})"
        )
    
    results = {}
    for name, smiles in sample_polymers.items():
        try:
            test_counter["count"] += 1
            predictions = predictor.predict_single(smiles)
            results[name] = {
                "smiles": smiles,
                "predictions": predictions
            }
        except Exception as e:
            test_counter["count"] = max(0, test_counter["count"] - 1)
            results[name] = {"error": str(e)}
    
    return {
        "results": results,
        "total_tested": len([r for r in results.values() if "predictions" in r]),
        "test_count": test_counter["count"],
        "remaining_tests": max(0, test_counter["limit"] - test_counter["count"])
    }

# Optional auth endpoints (still available but not required)
@app.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    """Register new user - OPTIONAL."""
    # Always return success for testing
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": "test_user", "email": user_data.email},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id="test_user",
        expires_at=datetime.utcnow() + access_token_expires
    )

@app.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    """Login user - OPTIONAL."""
    # Always return success for testing
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": "test_user", "email": user_data.email},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id="test_user",
        expires_at=datetime.utcnow() + access_token_expires
    )

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        full_name="Test User"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)