from PIL import Image
import io
import base64
import numpy as np
import cv2
import time



# creare a random numpy array of RGB values, 0-255
# arr = 255 * numpy.random.rand(20, 20, 3)
arr = cv2.imread("./car.jpg")


def array_to_base64(image):
    im = Image.fromarray(image.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    return base64.b64encode(rawBytes.read()).decode("utf-8")


def base64_to_array(base64_string):
    decoded = base64.b64decode(base64_string.encode("utf-8"))
    return np.array(Image.open(io.BytesIO(decoded)))



if __name__ == "__main__":
    st = time.time()
    img = base64_to_array(array_to_base64(arr))
    tt = time.time() - st

    cv2.imshow("Image", img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

    print(f"time taken: {tt} seconds")