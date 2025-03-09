from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
MODEL_PATH = "model/sign_language_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Class labels
CLASS_LABELS = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  
    image = image.resize((128, 128))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# WebSocket for ISL-to-Text
@app.websocket("/ws/predict")
async def websocket_isl_to_text(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes))

            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            predicted_label = CLASS_LABELS[predicted_class] if predicted_class < len(CLASS_LABELS) else "Unknown"

            await websocket.send_text(predicted_label)

    except WebSocketDisconnect:
        print("Client disconnected from ISL-to-Text WebSocket.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=True)
