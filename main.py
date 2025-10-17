import os
from dotenv import load_dotenv
import base64
from datetime import date, datetime, timezone
from typing import Optional, List, cast

import requests


from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request, APIRouter
# from fastapi.security import OAuth2PasswordBearer # REMOVED: No longer needed for Swagger UI auth flow

from sqlmodel import Field, SQLModel, create_engine, Session, select

import firebase_admin
from firebase_admin import credentials, auth, storage

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

# --- 1. Configuration & Environment Setup ---
load_dotenv("calorie.env")

class Settings:
    """Centralized class for application settings."""
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./calorie_app.db")
    FIREBASE_STORAGE_BUCKET: Optional[str]= os.getenv("FIREBASE_STORAGE_BUCKET")
    CLARIFAI_API_KEY: Optional[str] = os.getenv("CLARIFAI_API_KEY")
    CLARIFAI_USER_ID: str = "clarifai"
    CLARIFAI_APP_ID: str = "main"
    CLARIFAI_FOOD_MODEL_ID: str = "food-item-recognition"
    EDAMAM_APP_ID: Optional[str] = os.getenv("EDAMAM_APP_ID")
    EDAMAM_APP_KEY: Optional[str] = os.getenv("EDAMAM_APP_KEY")
    EDAMAM_NUTRITION_API_URL: str = "https://api.edamam.com/api/nutrition-details"
    ENV: str = os.getenv("ENV", "development") # 'production' or 'development'

settings = Settings()

def validate_settings_envs(s: Settings) -> None:
    missing = []
    if s.CLARIFAI_API_KEY is None:
        missing.append("CLARIFAI_API_KEY")
    if s.EDAMAM_APP_ID is None:
        missing.append("EDAMAM_APP_ID")
    if s.EDAMAM_APP_KEY is None:
        missing.append("EDAMAM_APP_KEY")
    if missing:
        raise RuntimeError(f"Missing is required for the environment variables: {','.join(missing)}")

validate_settings_envs(settings)


# --- 2. Firebase Initialization ---
def initialize_firebase():
    """Initializes Firebase Admin SDK."""
    cred_path = "serviceAccountKey.json"
    if not os.path.exists(cred_path):
        print(f"Error: Firebase service account Key not found at {cred_path}")
        print("Please ensure 'serviceAccountKey.json' is in the root directory of the project.")
        exit(1)

    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {'storageBucket': settings.FIREBASE_STORAGE_BUCKET})
        print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")
        exit(1)

# --- 3. Database Setup (SQLModel) ---
engine = create_engine(settings.DATABASE_URL, echo=True)

def create_db_and_tables():
    """Creates all database tables defined by SQLModel metadata."""
    SQLModel.metadata.create_all(engine)

def get_session():
    """Dependency to get a database session."""
    with Session(engine) as session:
        yield session

# --- 4. Database Models ---
class UserProfile(SQLModel, table=True):
    uid: str = Field(primary_key=True, index=True)
    display_name: Optional[str] = None
    email: Optional[str] = Field(default=None, index=True)
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    gender: Optional[str] = None
    activity_level: Optional[str] = None
    target_weight_kg: Optional[float] = None
    calorie_goal: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) # Added for clarity

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
    image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ActivityLogEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="userprofile.uid", index=True)
    activity_name: str
    duration_minutes: int
    calories_burned: int
    log_date: date = Field(default_factory=date.today, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# --- 5. External API Clients ---

class ClarifaiClient:
    """Client for interacting with the Clarifai API."""
    def __init__(self, api_key: str, user_id: str, app_id: str, model_id: str):
        if not api_key:
            raise ValueError("Clarifai API Key is not set.")
        self.api_key = api_key
        self.user_id = user_id
        self.app_id = app_id
        self.model_id = model_id
        self.channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(self.channel)

    async def analyze_food_image(self, image_bytes: bytes) -> dict:
        """Analyzes an image using Clarifai's food recognition model."""
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        metadata = (('authorization', 'Key ' + self.api_key),) # Note the space after 'Key'

        post_model_outputs_response = self.stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=resources_pb2.UserAppIDSet(user_id=self.user_id, app_id=self.app_id),
                model_id=self.model_id,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(base64=base64_image) #type: ignore
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

        food_concepts = []
        if post_model_outputs_response.outputs:
            for concept in post_model_outputs_response.outputs[0].data.concepts:
                food_concepts.append({"name": concept.name, "value": concept.value})
        return {"food_detections": food_concepts}

clarifai_client = ClarifaiClient(
    api_key=cast(str,settings.CLARIFAI_API_KEY), #type: ignore
    user_id=settings.CLARIFAI_USER_ID,
    app_id=settings.CLARIFAI_APP_ID,
    model_id=settings.CLARIFAI_FOOD_MODEL_ID
)

class EdamamClient:
    """Client for interacting with the Edamam Nutrition Analysis API."""
    def __init__(self, app_id: str, app_key: str, api_url: str):
        if not app_id or not app_key:
            raise ValueError("Edamam API credentials are not set.")
        self.app_id = app_id
        self.app_key = app_key
        self.api_url = api_url

    async def get_nutrition_data(self, food_text: str) -> dict:
        """Fetches nutrition data for a given food item from Edamam."""
        headers = {'Content-Type': 'application/json'}
        params = {'app_id': self.app_id, 'app_key': self.app_key}
        data = {"ingredients": [food_text]}

        try:
            response = requests.post(self.api_url, headers=headers, params=params, json=data, timeout=10)
            response.raise_for_status()
            nutrition_data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Edamam API request failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve nutrition data from Edamam: {e}"
            )

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

edamam_client = EdamamClient(
    app_id=cast(str,settings.EDAMAM_APP_ID), #type: ignore
    app_key=cast(str, settings.EDAMAM_APP_KEY), #type: ignore
    api_url=settings.EDAMAM_NUTRITION_API_URL
)

# --- 6. FastAPI App Initialization & Authentication ---
app = FastAPI(
    title="Calorie & Fitness Tracker Backend",
    description="API for managing user profiles, food logs, and activity logs with Firebase Auth, SQLModel, Clarifai, and Edamam.",
    version="0.1.0",
    # REMOVED: No openapi_extra for securitySchemes here
    # This prevents the "Authorize" button from appearing in Swagger UI
)

# MODIFIED: get_current_user_id no longer depends on oauth2_scheme
async def get_current_user_id(
    request: Request, # Get the raw request object
    # token: str = Depends(oauth2_scheme) # REMOVED: No longer using oauth2_scheme
) -> str:
    """
    Dependency to get the current user's UID from a Firebase ID token.
    Allows a mock UID for local development.
    The client should send the Firebase ID token in the Authorization header as "Bearer <token>".
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Please provide a Firebase ID Token.",
            headers={"WWW-Authenticate": "Bearer"}, # Still good practice to suggest Bearer
        )

    # Extract the token part, assuming "Bearer <token>" format
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth_header.replace("Bearer ", "")

    if token == "MOCK_USER_UID" and settings.ENV != "production":
        print("Using MOCK_USER_UID for development.")
        return "mock_user_123"

    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- 7. API Routes ---

# Router for user-related endpoints
user_router = APIRouter(prefix="/users", tags=["Users"])

@user_router.post("/profile", response_model=UserProfile)
async def create_or_update_user_profile(
    profile_data: UserProfile,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """Create or update a user's fitness profile."""
    if profile_data.uid != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User ID in profile does not match authenticated user."
        )

    existing_profile = session.get(UserProfile, current_user_id)
    if existing_profile:
        for key, value in profile_data.dict(exclude_unset=True).items():
            setattr(existing_profile, key, value)
        existing_profile.updated_at = datetime.now(timezone.utc)
        session.add(existing_profile)
        session.commit()
        session.refresh(existing_profile)
        return existing_profile
    else:
        new_profile = UserProfile.from_orm(profile_data)
        session.add(new_profile)
        session.commit()
        session.refresh(new_profile)
        return new_profile

@user_router.get("/profile/{user_id}", response_model=UserProfile)
async def get_user_profile(
    user_id: str,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """Retrieve a user's fitness profile."""
    if user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own profile."
        )

    profile = session.get(UserProfile, user_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")
    return profile

# Router for food-related endpoints
food_router = APIRouter(prefix="/food", tags=["Food Logs & Recognition"])

@food_router.post("/logs", response_model=FoodLogEntry, status_code=status.HTTP_201_CREATED)
async def add_food_log(
    entry: FoodLogEntry,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """Add a new food log entry for the current user."""
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

@food_router.get("/logs/{log_date}", response_model=List[FoodLogEntry])
async def get_food_logs_for_date(
    log_date: date,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """Retrieve all food logs for a specific date for the current user."""
    statement = select(FoodLogEntry).where(
        FoodLogEntry.user_id == current_user_id,
        FoodLogEntry.log_date == log_date
    )
    logs = session.exec(statement).all()
    return logs

@food_router.post("/recognition/upload", response_model=dict)
async def upload_food_image_for_recognition(
    file: UploadFile = File(...),
    current_user_id: str = Depends(get_current_user_id)
):
    """
    Upload a food image for AI recognition (Clarifai) and store it in Firebase Storage.
    Returns detected food items and the public URL of the uploaded image.
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only image files are allowed.")
    if file.size and file.size > 5 * 1024 * 1024:
         raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image file too large. Max 5MB.")

    image_bytes = await file.read()

    clarifai_result = await clarifai_client.analyze_food_image(image_bytes)

    file_extension = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
    filename = f"food_images/{current_user_id}/{datetime.now(timezone.utc).isoformat().replace(':', '-').replace('.', '_')}_{file.filename}"

    try:
        bucket = storage.bucket()
        blob = bucket.blob(filename)
        blob.upload_from_string(image_bytes, content_type=file.content_type)
        blob.make_public()
        image_public_url = blob.public_url
    except Exception as e:
        print(f"Error uploading image to Firebase Storage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload image to storage: {e}"
        )

    clarifai_result["image_url"] = image_public_url
    return clarifai_result

@food_router.get("/nutrition/search", response_model=dict)
async def search_food_nutrition(
    food_query: str,
    current_user_id: str = Depends(get_current_user_id) # Authenticate, though not strictly needed for this GET
):
    """Search for nutrition data for a given food item using Edamam."""
    nutrition_data = await edamam_client.get_nutrition_data(food_query)
    if nutrition_data["calories"] == 0 and nutrition_data["serving_size_g"] == 0 and nutrition_data["protein_g"] == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Nutrition data not found for '{food_query}'")
    return nutrition_data

# Router for activity-related endpoints
activity_router = APIRouter(prefix="/activities", tags=["Activity Logs"])

@activity_router.post("/logs", response_model=ActivityLogEntry, status_code=status.HTTP_201_CREATED)
async def add_activity_log(
    entry: ActivityLogEntry,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """Add a new activity log entry for the current user."""
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

@activity_router.get("/logs/{log_date}", response_model=List[ActivityLogEntry])
async def get_activity_logs_for_date(
    log_date: date,
    current_user_id: str = Depends(get_current_user_id),
    session: Session = Depends(get_session)
):
    """Retrieve all activity logs for a specific date for the current user."""
    statement = select(ActivityLogEntry).where(
        ActivityLogEntry.user_id == current_user_id,
        ActivityLogEntry.log_date == log_date
    )
    logs = session.exec(statement).all()
    return logs

# Add routers to the main FastAPI app
app.include_router(user_router)
app.include_router(food_router)
app.include_router(activity_router)

@app.on_event("startup")
async def startup_event():
    """Actions to perform when the application starts."""
    initialize_firebase()
    create_db_and_tables()
    print("Application startup complete.")

@app.get("/")
async def read_root():
    """Root endpoint for the API."""
    return {"message": "Welcome to the Calorie & Fitness Tracker API!"}

# --- 8. Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    # If running directly, initialize Firebase and create tables
    # (startup_event will also do this, but this ensures it for direct run)
    initialize_firebase()
    create_db_and_tables()
    uvicorn.run(app, host="0.0.0.0", port=8000)