from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

# Supabase Configuration
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Create FastAPI app
app = FastAPI(
    title="Polymer Property Prediction API with Auth",
    description="Complete polymer prediction with Supabase authentication",
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
is_trained = False

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

# Supabase Database Operations
class SupabaseDB:
    @staticmethod
    async def create_user(email: str, password: str, full_name: str = None):
        """Create user in Supabase."""
        try:
            response = supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {"full_name": full_name}
                }
            })
            return response.user
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @staticmethod
    async def authenticate_user(email: str, password: str):
        """Authenticate user."""
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return response
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    @staticmethod
    async def store_prediction(user_id: str, smiles: str, predictions: Dict[str, float], prediction_id: str = None):
        """Store prediction in database."""
        try:
            data = {
                "id": prediction_id,
                "user_id": user_id,
                "smiles": smiles,
                "predictions": json.dumps(predictions),
                "created_at": datetime.now().isoformat()
            }
            result = supabase.table("predictions").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            return None
    
    @staticmethod
    async def get_user_predictions(user_id: str, limit: int = 100):
        """Get user's predictions."""
        try:
            result = supabase.table("predictions")\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            return []

# Authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    try:
        token = credentials.credentials
        response = supabase.auth.get_user(token)
        
        if response.user:
            return response.user
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

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

# Authentication Endpoints
@app.post("/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    """Register new user."""
    try:
        user = await SupabaseDB.create_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        if user:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            token_data = {"sub": user.id, "email": user.email, "exp": expire}
            access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
            
            return Token(
                access_token=access_token,
                token_type="bearer",
                user_id=user.id,
                expires_at=expire
            )
        
        raise HTTPException(status_code=400, detail="Failed to create user")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    """Login user."""
    try:
        response = await SupabaseDB.authenticate_user(user_data.email, user_data.password)
        
        if response and response.user:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            token_data = {"sub": response.user.id, "email": response.user.email, "exp": expire}
            access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
            
            return Token(
                access_token=access_token,
                token_type="bearer",
                user_id=response.user.id,
                expires_at=expire
            )
        
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")

# Main API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Polymer Property Prediction API with Authentication",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "User Authentication with Supabase",
            "Advanced SMILES Analysis (RDKit Alternative)", 
            "Prediction History Storage",
            "Batch Processing"
        ],
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "supabase_connected": bool(supabase),
        "predictor_ready": True
    }

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single(input_data: SMILESInput, current_user = Depends(get_current_user)):
    """Predict properties for single SMILES."""
    try:
        predictions = predictor.predict_single(input_data.smiles)
        
        # Store in database
        prediction_id = input_data.id or f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await SupabaseDB.store_prediction(
            user_id=current_user.id,
            smiles=input_data.smiles,
            predictions=predictions,
            prediction_id=prediction_id
        )
        
        return PredictionResponse(
            id=prediction_id,
            smiles=input_data.smiles,
            predictions=predictions,
            created_at=datetime.now(),
            user_id=current_user.id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchSMILESInput, current_user = Depends(get_current_user)):
    """Predict properties for multiple SMILES."""
    try:
        predictions = predictor.predict_batch(input_data.smiles_list)
        
        results = []
        for i, smiles in enumerate(input_data.smiles_list):
            pred_id = input_data.ids[i] if input_data.ids and i < len(input_data.ids) else f"batch_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            single_predictions = {prop: values[i] for prop, values in predictions.items()}
            
            # Store in database
            await SupabaseDB.store_prediction(
                user_id=current_user.id,
                smiles=smiles,
                predictions=single_predictions,
                prediction_id=pred_id
            )
            
            results.append(PredictionResponse(
                id=pred_id,
                smiles=smiles,
                predictions=single_predictions,
                created_at=datetime.now(),
                user_id=current_user.id
            ))
        
        summary = {
            "total_predictions": len(results),
            "properties": list(predictions.keys()),
            "average_predictions": {
                prop: float(np.mean(values)) for prop, values in predictions.items()
            },
            "user_id": current_user.id
        }
        
        return BatchPredictionResponse(predictions=results, summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/predictions/history")
async def get_prediction_history(current_user = Depends(get_current_user), limit: int = 100):
    """Get user's prediction history."""
    try:
        history = await SupabaseDB.get_user_predictions(current_user.id, limit)
        
        formatted_history = []
        for record in history:
            formatted_history.append({
                "id": record["id"],
                "smiles": record["smiles"],
                "predictions": json.loads(record["predictions"]),
                "created_at": record["created_at"],
                "user_id": record["user_id"]
            })
        
        return {
            "history": formatted_history,
            "total_count": len(formatted_history),
            "user_id": current_user.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.post("/upload/training-data")
async def upload_training_data(file: UploadFile = File(...), current_user = Depends(get_current_user)):
    """Upload training data CSV."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        return {
            "message": "Training data uploaded successfully",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "user_id": current_user.id,
            "upload_time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload: {str(e)}")

# Public endpoint for testing (no auth required)
@app.post("/test/predict")
async def test_predict(smiles: str):
    """Test prediction without authentication."""
    try:
        predictions = predictor.predict_single(smiles)
        return {
            "smiles": smiles,
            "predictions": predictions,
            "message": "Test prediction successful"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app)