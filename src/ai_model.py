import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision  
from PIL import Image



def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels'], pred_scores



def draw_boxes(boxes, classes, labels, image, scores):
    # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = (0, 255, 0) # green
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 3
        )
        cv2.putText(image, f"{classes[i].upper()} {scores[i]:.3f}", (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=3, 
                    lineType=cv2.LINE_AA)
    return image



def run_object_detection(image):
    # read the image and run the inference for detections
    boxes, classes, labels, scores = predict(image, model, device, 0.7)
    image = draw_boxes(boxes, classes, labels, image, scores)
    return image


coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])


# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# load the model
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)

# load the model on to the computation device
model.eval().to(device)


if __name__ == "__main__":

    final_image = run_object_detection(cv2.imread("./car.jpg"))

    cv2.imshow("Image", final_image)

    # waits for user to press any key 
    # (this is necessary to avoid Python kernel form crashing) 
    cv2.waitKey(0) 
    
    # closing all open windows 
    cv2.destroyAllWindows()