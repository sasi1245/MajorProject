from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import random
import uvicorn

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

# Define dataset folder for ISL images
ISL_MEDIA_FOLDER = "dataset"
os.makedirs(ISL_MEDIA_FOLDER, exist_ok=True)

# Function to load ISL dictionary
def load_isl_dictionary():
    isl_dict = {}
    for category in os.listdir(ISL_MEDIA_FOLDER):
        category_path = os.path.join(ISL_MEDIA_FOLDER, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]
            if images:
                isl_dict[category.lower()] = images
    return isl_dict

# Load dictionary
isl_dict = load_isl_dictionary()

# WebSocket for Text-to-ISL
@app.websocket("/ws/convert")
async def websocket_text_to_isl(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            text = text.lower()

            if text in isl_dict:
                selected_image = random.choice(isl_dict[text])
                media_url = f"http://localhost:5002/static/isl_media/{text}/{selected_image}"
                await websocket.send_text(media_url)
            else:
                await websocket.send_text("ISL representation not found")

    except WebSocketDisconnect:
        print("Client disconnected from Text-to-ISL WebSocket.")

# Serve static ISL images
from fastapi.staticfiles import StaticFiles
app.mount("/static/isl_media", StaticFiles(directory=ISL_MEDIA_FOLDER), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002, reload=True)
