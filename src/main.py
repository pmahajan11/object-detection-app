from fastapi import FastAPI
from fastapi.params import Body
from fastapi.middleware.cors import CORSMiddleware
from ai_model import run_object_detection
from image_encode_decode import array_to_base64, base64_to_array
from pydantic import BaseModel
import time



app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EncodedImage(BaseModel):
    encoded_image: str


@app.get("/")
def root():
    return {"message": "Object Detection App"}


@app.post("/detect-objects")
def detect_objects(payload: EncodedImage):
    st1 = time.time()
    image_array = base64_to_array(payload.encoded_image)
    image_with_boxes = run_object_detection(image_array)
    encoded_image_with_boxes = array_to_base64(image_with_boxes)
    response_time = time.time() - st1
    return {
        "encoded_image": encoded_image_with_boxes,
        "response_time": response_time
    }

