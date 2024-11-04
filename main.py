from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from websockets.exceptions import ConnectionClosed
import cv2
from internal.detector import FallDetector
from reader import Camera

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory="templates")


class VideoSource:
    video_source = "rtsp://admin:123456a%40@192.168.44.20"
    encode = "h264"
    camera_width = 1920
    camera_height = 1080


# Global variables
# video_capture = None
source = VideoSource()
detector = FallDetector('internal/yolov7_pose/weights/yolov7-w6-pose.pt')
# detector = FallDetector('internal/yolo11n-pose.pt')

# Mount static files directory (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global source

    await websocket.accept()
    cam_reader = Camera(source.video_source)
    try:
        while cam_reader.isOpened():
            try:
                frame = cam_reader.getFrame()
                if frame is None:
                    break

                frame = detector.process_image(frame)

                # Encode frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                buffer = jpeg.tobytes()
                await websocket.send_bytes(buffer)
            except Exception as e:
                print(f"Websocket Error: {e}")
                break
    except (WebSocketDisconnect, ConnectionClosed):
        print('Client disconnected')

    print('Done')


@app.get("/start_stream")
async def start_stream(rtsp_url: str):
    # global video_capture
    global source
    source.video_source = rtsp_url


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
