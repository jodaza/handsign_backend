from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from model_loader import HandSignPredictor

# Instanciar el predictor
predictor = HandSignPredictor("model.p")

app = FastAPI()
print("Cargando modelo...")

# Permitir frontend desde cualquier origen (puedes restringir luego)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Cliente conectado")
    await websocket.accept()

    while True:
        try:
            data = await websocket.receive_json()
            image_base64 = data["image"]
            letra = predictor.predict(image_base64)
            await websocket.send_json({"letra": letra})
        except Exception as e:
            print("Error:", e)
            break