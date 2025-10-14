import os 
from dotenv import load_dotenv 
import requests # for the Edamam API 
import json #For Clarifai JSON parsing 
import base64 #for Clarifai image encoding 

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel 
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

from fastapi import FastAPI, Depends, HTTPException, status, Form, UploadFile, File, Request
from fastapi.security import OAuth2PasswordBearer
from typing import Optional, List 
from datetime import date, datetime, timezone, timedelta

#-- Databse Imports (SQL Alchemy) --#

from sqlmodel import Field, SQLModel, create_engine, Session, select

#Initializing the envirement variables 

load_dotenv()

#Firebase Admin SDK 

import firebase_admin 
from firebase_admin import credentials, auth, storage 

try:
    #Looks for a securityAccountKey.json file in the root directory 
    cred_path = "serviceAccountKey.json"
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Firebase service account Key not found at {"cred path"}")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred,{
        'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
    })
    print("Firebase Admin SDK initialized successfully.")
except FileNotFoundError as fnf_e:
    print(f"Error:{fnf_e}")
    print("Please ensure 'serviceAccountKey.json' is in the root directory of the project.")
    #Exit if the file is not found 
    exit(1)
except Exception as e:
    print (f"Error initializing Firebase Admin SDK: {e}")
    #Exit if critical dependency not met 
    exit(1)


#Database SetUp (SQL Model)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calorie_app.db")
engine = create_engine(DATABASE_URL, echo=True) #echo=True for SQL logging 

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

#Dependency to get a database session 

def get_session():
    with Session(engine) as session:
        yield session

#Models for the SQLModel Databse 

class UserProfile(SQLModel, table=True):
    #uid is the primary key from firebase 
    uid: str = Field(primary_key=True, index=True)
    display_name: Optional[str] = None 
    email: Optional[str] = Field(default=None, index=True) #I'll need this for querying 
    age: Optional[int] =None 
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None 
    gender: Optional[str] = None 
    activity_level: Optional[str] = None 
    target_weight_kg: Optional[float] = None 
    calorie_goal: Optional[float] = None 
    created_at: datetime = Field(default_factory= lambda: datetime.now(timezone.utc))
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class FoodLogEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="userprofile.uid", index=True)
    food_name: str
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    serving_size_g: Optional[float] = None
    meal_type: Optional[str] = None
    log_date: date = Field(default_factory=date.today, index=True)
    image_url: Optional[str] = None # URL to the uploaded food image
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ActivityLogEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field (foreign_key="userprofile.uid", index=True)
    activity_name: str
    duration_minutes: int
    calories_burned: int
    log_date: date = Field(default_factory=date.today, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    #External API integration ( Clarifai )

CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY")
CLARIFAI_FOOD_MODEL_ID = "food-item-recognition"
CLARIFAI_USER_ID = "clarifai" #Most Clarifai public models are under 'clarifai' user 
CLARIFAI_APP_ID = "main" #Most Clarifai public models are under 'main' app

channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub( channel )

async def analyze_food_image(image_bytes: bytes) -> dict:
    if not CLARIFAI_API_KEY:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail = "Clarifai API Key not set. ")
    
    #Encode image to base64
    base64_image= base64.b64encode( image_bytes ). decode('utf-8')

    metadata = (('authorization', 'Key' + CLARIFAI_API_KEY),)
    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id = resources_pb2.UserAppIDSet(user_id=CLARIFAI_USER_ID, app_id=CLARIFAI_APP_ID),
            MODEL_ID=CLARIFAI_FOOD_MODEL_ID,  # type: ignore
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=base64_image # type: ignore
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(f"Clarifai API call failed: {post_model_outputs_response.status.description}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clarifai image analysis failed: {post_model_outputs_response.status.description}"
        )

    # Extract relevant concepts/foods
    food_concepts = []
    if post_model_outputs_response.outputs:
        for concept in post_model_outputs_response.outputs[0].data.concepts:
            food_concepts.append({"name": concept.name, "value": concept.value})

    return {"food_detections": food_concepts}

#Edamam API ( for nutrition data )

EDAMAM_APP_ID = os.getenv("EDAMAM_APP_ID")
EDAMAM_APP_KEY = os.getenv("EDAMAM_APP_KEY")
EDAMAM_API_URL = "https://api.edamam.com/api/nutrition-details"

async def get_nutition_data_from_edamam(food_text:str) -> dict:
    if not EDAMAM_APP_ID or not EDAMAM_APP_KEY:
        raise 
    HTTPException (status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                   detail="Edamam API credentials not set.")
    
    headers = {'Content-Type': 'application/json'}
    params = {'app_id':EDAMAM_APP_ID, 'app_key':EDAMAM_APP_KEY}

    #At this point edamam expects a list of ingredients 
    #For a simple food_text query , we can send it as one ingredient 

    data = {"ingredients": [food_text]}

    try:
        response = requests.post(EDAMAM_NURITION_API_URL, headers = headers , params= params , json=data, timeout=10) #type:ignore
        response.raise_for_status()#Raise an exception for HTTP errors (4xx or 5xx)
        nutrition_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Edamam API request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve nutrition data from Edamam: {e}"
        )
    
    # Process Edamam response
    if 'totalNutrients' in nutrition_data:
        calories = nutrition_data.get('calories', 0)
        protein_g = nutrition_data['totalNutrients'].get('PROCNT', {}).get('quantity', 0)
        carbs_g = nutrition_data['totalNutrients'].get('CHOCDF', {}).get('quantity', 0)
        fat_g = nutrition_data['totalNutrients'].get('FAT', {}).get('quantity', 0)
        
        return {
            "calories": calories,
            "protein_g": round(protein_g, 2),
            "carbs_g": round(carbs_g, 2),
            "fat_g": round(fat_g, 2),
            "serving_size_g": nutrition_data.get('totalWeight', 0)
        }
    
    return {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "serving_size_g": 0}


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Calorie & Fitness Tracker Backend",
    description="API for managing user profiles, food logs, and activity logs with Firebase Auth, SQLModel, Clarifai, and Edamam.",
    version="0.1.0",
)

# --- Dependency for Firebase Authentication ---
# Expects a Bearer token in the Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl is just for Swagger UI documentation

async def get_current_user_id(
    request: Request,
    token: str = Depends(oauth2_scheme)
):
    """
    Dependency to get the current user's UID from a Firebase ID token
    provided in the Authorization header (Bearer token).
    For local development, it accepts a special "MOCK_USER_UID" string.
    """
    # Allow a mock UID for local development without needing a real token
    if token == "MOCK_USER_UID" and os.getenv("ENV") != "production": # Add ENV var to prevent in prod
        print("Using MOCK_USER_UID for development.")
        return "mock_user_123" # A fake UID for testing

    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        return uid
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- Routes ---

@app.on_event("startup")
def on_startup():
    print("Creating database tables...")
    create_db_and_tables()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Calorie & Fitness Tracker API!"}

# --- User Profile Endpoints ---

@app.post("/users/profile", response_model=UserProfile)
async def create_or_update_user_profile(
    profile_data: UserProfile, # Renamed to avoid conflict with `profile` instance
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """
    Create or update a user's fitness profile.
    The 'uid' in the profile must match the authenticated user's ID.
    """
    if profile_data.uid != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User ID in profile does not match authenticated user."
        )
    
    # Check if profile already exists
    existing_profile = session.get(UserProfile, current_user_id)

    if existing_profile:
        # Update existing profile
        for key, value in profile_data.dict(exclude_unset=True).items():
            setattr(existing_profile, key, value)
        existing_profile.updated_at = datetime.now(timezone.utc) # Update timestamp
        session.add(existing_profile)
        session.commit()
        session.refresh(existing_profile)
        return existing_profile
    else:
        # Create new profile
        new_profile = UserProfile.from_orm(profile_data)
        session.add(new_profile)
        session.commit()
        session.refresh(new_profile)
        return new_profile

@app.get("/users/profile/{user_id}", response_model=UserProfile)
async def get_user_profile(
    user_id: str,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """
    Retrieve a user's fitness profile.
    Only allows a user to fetch their own profile.
    """
    if user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own profile."
        )
    
    profile = session.get(UserProfile, user_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")
    return profile

# --- Food Logging Endpoints ---

@app.post("/food-logs", response_model=FoodLogEntry, status_code=status.HTTP_201_CREATED)
async def add_food_log(
    entry: FoodLogEntry,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """
    Add a new food log entry for the current user.
    """
    if entry.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot add food log for another user."
        )
    
    db_entry = FoodLogEntry.from_orm(entry)
    session.add(db_entry)
    session.commit()
    session.refresh(db_entry)
    return db_entry

@app.get("/food-logs/{log_date}", response_model=List[FoodLogEntry])
async def get_food_logs_for_date(
    log_date: date,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """
    Retrieve all food logs for a specific date for the current user.
    """
    statement = select(FoodLogEntry).where(
        FoodLogEntry.user_id == current_user_id,
        FoodLogEntry.log_date == log_date
    )
    logs = session.exec(statement).all()
    return logs

# --- Food Recognition and Nutrition Endpoints ---

@app.post("/food-recognition/upload", response_model=dict)
async def upload_food_image_for_recognition(
    file: UploadFile = File(...),
    current_user_id: str = Depends(get_current_user_id)
):
    """
    Upload a food image for AI recognition (Clarifai) and store it in Firebase Storage.
    Returns detected food items and the public URL of the uploaded image.
    """
    if not file.content_type.startswith('image/'): #type:ignore
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only image files are allowed.")
    if file.size > 5 * 1024 * 1024: # Limit to 5MB #type:ignore
         raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image file too large. Max 5MB.")

    image_bytes = await file.read()
    
    # 1. Analyze image with Clarifai
    clarifai_result = await analyze_food_image(image_bytes)

    # 2. Upload image to Firebase Storage
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg" #type:ignore
    # Ensure unique filename to prevent overwrites, include user ID
    filename = f"food_images/{current_user_id}/{datetime.now(timezone.utc).isoformat().replace(':', '-').replace('.', '_')}_{file.filename}"
    
    try:
        bucket = storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_string(image_bytes, content_type=file.content_type) # type: ignore
        blob.make_public() # Make the image publicly accessible
        image_public_url = blob.public_url
    except Exception as e:
        print(f"Error uploading image to Firebase Storage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload image to storage: {e}"
        )
    
    clarifai_result["image_url"] = image_public_url

    return clarifai_result

@app.get("/food-nutrition/search", response_model=dict) # Response model can be more specific
async def search_food_nutrition(
    food_query: str,
    current_user_id: str = Depends(get_current_user_id) # Ensure user is authenticated
):
    """
    Search for nutrition data for a given food item using Edamam.
    """
    nutrition_data = await get_nutition_data_from_edamam(food_query)
    if nutrition_data["calories"] == 0 and nutrition_data["serving_size_g"] == 0: # More robust check if data was found
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Nutrition data not found for '{food_query}'")
    return nutrition_data

# --- Activity Logging Endpoints ---

@app.post("/activity-logs", response_model=ActivityLogEntry, status_code=status.HTTP_201_CREATED)
async def add_activity_log(
    entry: ActivityLogEntry,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """
    Add a new activity log entry for the current user.
    """
    if entry.user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot add activity log for another user."
        )
    
    db_entry = ActivityLogEntry.from_orm(entry)
    session.add(db_entry)
    session.commit()
    session.refresh(db_entry)
    return db_entry

@app.get("/activity-logs/{log_date}", response_model=List[ActivityLogEntry])
async def get_activity_logs_for_date(
    log_date: date,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """
    Retrieve all activity logs for a specific date for the current user.
    """
    statement = select(ActivityLogEntry).where(
        ActivityLogEntry.user_id == current_user_id,
        ActivityLogEntry.log_date == log_date
    )
    logs = session.exec(statement).all()
    return logs

# --- Health Metrics/Goals (Future Expansion) ---
# You could add endpoints for tracking weight over time, setting goals,
# calculating BMR/TDEE, etc.

if __name__ == "__main__":
    import uvicorn
    # Create tables on startup if running directly
    create_db_and_tables() 
    uvicorn.run(app, host="0.0.0.0", port=8000)



