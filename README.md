# Face-mask-Detection
This project aims to detect whether a person is wearing a mask or not using OpenCV, a popular computer vision library. The system detects faces in a live video stream and analyzes if the face is covered with a mask or not. It provides real-time feedback on the video feed, marking faces with or without masks.

## Requirements
- Python 3.x
- OpenCV
- Numpy
### Install the required libraries using pip:
```bash
pip install opencv-python numpy
```
## How to Run
### Clone the repository or download the files.
```bash
git clone https://github.com/An1rud/Face-mask-Detection.git
```
- Ensure that the haarcascade files for face, eyes, mouth, and upper body are in the data/xml/ directory.
- You can download these cascades from OpenCV GitHub repository.
### Run the mask_detection.py script:
```bash
python mask_detection.py
```
- A video window will open showing the live webcam feed with mask detection results.
- White rectangles indicate detected faces.
- "Face Detected with mask" will be displayed when a face is detected but no mouth is detected (assumed to be wearing a mask).
- "Face Detected without mask" will be displayed when a face and mouth are detected together (assumed to not be wearing a mask).
## Adjustments
- You can adjust the bw_threshold variable to fine-tune the black and white conversion threshold based on lighting conditions.
- The script currently uses a single webcam (cap = cv2.VideoCapture(0)), but you can modify this to use a different video source.
## Important Notes
- Ensure proper lighting conditions for accurate detection.
- This is a simple mask detection system and may not be suitable for all scenarios. It serves as a basic demonstration of face and object detection in OpenCV.
