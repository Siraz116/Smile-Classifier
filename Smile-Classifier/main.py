import os
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request, Depends, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image
import uvicorn

from sqlalchemy.orm import Session

from crud import create_image
# Local imports
from database import engine, SessionLocal, Base
from models import Classification
from classifier import SmileClassifier
from fastapi.staticfiles import StaticFiles

import schemas, crud
import tensorflow.keras.models as keras
import numpy as np


model = keras.load_model('my_model_smile_or_not.h5')

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try : 
        yield db
    finally:
        db.close()


# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

# Load pre-trained model
# model_data = SmileClassifier.load_model('smile_classifier.pkl')
# classifier = model_data['model']
# scaler = model_data['scaler']

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with model explanation"""
    context = {
        "request": request,
        "model_name": "Random Forest Classifier",
        "dataset": "Smiling/Not Smiling Faces",
        "key_features": [
            "Grayscale image conversion",
            "Image resizing to 64x64",
            "Feature scaling",
            "Ensemble learning"
        ]
    }
    return templates.TemplateResponse("home.html", context)

@app.get("/classify", response_class=HTMLResponse)
async def classify_page(request: Request):
    """Classify page with file upload"""
    context = {"request": request}
    return templates.TemplateResponse("classify.html", context)

@app.post("/classify")
async def classify_image(file: UploadFile = File(...), db:Session=Depends(get_db)):
    """
    Classify uploaded image
    1. Save image
    2. Preprocess image
    3. Predict smile
    4. Save classification result
    """
    # Generate unique filename
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join("uploads", filename)

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    # Save uploaded file
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    with open(filepath, 'rb') as img_file:
        image = Image.open(img_file)
        
        # Resize and convert to RGB
        image = image.resize((64, 64))
        image = image.convert("RGB")  # Ensure 3 color channels

    # Convert to numpy array and reshape for the model
    img_array = np.array(image).reshape(1, 64, 64, 3) / 255.0  # Normalize pixel values
    
    # Make prediction
    prediction = model.predict(img_array)
    class_name = "Smiling" if prediction[0][0] > 0.5 else "Not Smiling"
    print(f"====TTT============Prediction: {class_name}==========TTTT")
        
    image = schemas.ImageCreate(image_path=filepath, classification=class_name)

    create_image(db, image)
    return RedirectResponse(url="/history", status_code=status.HTTP_302_FOUND)

@app.get("/history")
async def history(request: Request, db:Session=Depends(get_db)):
    """Display classification history"""

    images = crud.get_images(db)
    return templates.TemplateResponse("history.html", {"request": request, "classifications": images})




@app.post("/upload")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join("uploads", unique_filename)
    
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    # Open the saved image file 
    
    
    db_image = UploadedImage(
        filename=unique_filename,
        original_filename=file.filename,
        classification=result
        )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    
    return {"filename": unique_filename, "prediction": result}



if __name__ == "__main__":
    uvicorn.run(app="main:app", host="127.0.0.1", port=8000, reload=True)
