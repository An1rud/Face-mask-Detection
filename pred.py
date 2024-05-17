import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenetv2
from tensorflow.keras.models import load_model

def load_face_detector(model):
    if model == 'resnet50':
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNet(prototxtPath, weightsPath)
    else:
        raise ValueError("Unsupported face detection model")
    return net

def detect_face(frame, net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            return (startX, startY, endX, endY), face

    return None, None

def load_mask_detector(model):
    return load_model(model)

def preprocess_image(image, model):
    if model == 'resnet50':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image = preprocess_input_resnet50(image)
    elif model == 'mobilenetv2':
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        image = preprocess_input_mobilenetv2(image)
    else:
        raise ValueError("Unsupported model")
    return image

def predict_mask(image, face_detector, mask_detector, model):
    frame = image.copy()
    face_coords, face_image = detect_face(frame, face_detector)

    if face_coords is not None:
        face_image = preprocess_image(face_image, model)
        face_image = np.expand_dims(face_image, axis=0)
        (mask, withoutMask) = mask_detector.predict(face_image)[0]

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        (startX, startY, endX, endY) = face_coords

        # Increase text size for the label
        font_scale = 6.0  # Adjust the font scale to increase the size of the text
        thickness = 6
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        label_color = color  # Green color for label text

        cv2.putText(frame, label, (startX, startY - 10),
                    label_font, font_scale, label_color, thickness)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    return frame

if __name__ == "__main__":
    # Example usage:
    # Load models
    face_detector = load_face_detector("resnet50")
    mask_detector = load_mask_detector("mask_detector_resnet50.model")

    # Load image
    image = cv2.imread("mask.jpg")

    # Predict mask
    result = predict_mask(image, face_detector, mask_detector, "resnet50")

    # Resize the result image for display
    result = cv2.resize(result, (800, 600))

    font_scale = 1.5
    thickness = 2
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_color = (0, 255, 0)  # Green color for label text

    text = "Result"
    cv2.putText(result, text, (10, 30), label_font, font_scale, label_color, thickness)

    # Display result
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
