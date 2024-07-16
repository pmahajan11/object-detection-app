import streamlit as st
from PIL import Image
import io
import base64
import numpy as np
import time
import requests
import json


def array_to_base64(image):
    im = Image.fromarray(image.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    return base64.b64encode(rawBytes.read()).decode("utf-8")


def base64_to_array(base64_string):
    decoded = base64.b64decode(base64_string.encode("utf-8"))
    return np.array(Image.open(io.BytesIO(decoded)))


st.header("Object Detection App")

container = st.container()

picture = st.camera_input("Take a picture")

if picture:
    image = np.array(Image.open(picture))
    encoded_image = array_to_base64(image)
    data = json.dumps({"encoded_image": encoded_image})
    response = requests.request("POST", "http://127.0.0.1:8000/detect-objects", data=data)
    if response.status_code == 200:
        response_json = response.json()
        final_image = base64_to_array(response_json["encoded_image"])
        # container.write(f'Response time: {response_json["response_time"]:.3f} seconds')
        container.image(final_image)
    else:
        container.write("Something went wrong!")