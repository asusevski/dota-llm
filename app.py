import threading
import queue
import cv2
from PIL import Image
from transformers import pipeline
import mss
import numpy as np
from PIL import Image, ImageDraw
import torch
import cohere
import pyautogui
from ultralytics import YOLO

# Find the Dota 2 window (using the previous example)
import pygetwindow as gw
windows = gw.getWindowsWithTitle("Dota 2")
if windows:
    dota2_window = windows[0]
else:
    raise Exception("Dota 2 window not found")

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

def capture_dota2_window_opencv():
    with mss.mss() as sct:
        monitor = {
            "top": dota2_window.top,
            "left": dota2_window.left,
            "width": dota2_window.width,
            "height": dota2_window.height,
            "mon": 1,
        }
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

# Function to convert OpenCV image to Pillow
def opencv_to_pillow(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Function to convert Pillow image to OpenCV
def pillow_to_opencv(pillow_image):
    return cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)

# Function to draw bounding boxes on Pillow image
def draw_bounding_boxes(pillow_image, boxes):
    draw = ImageDraw.Draw(pillow_image)
    for box in boxes:
        draw.rectangle(box, outline="red", width=2)
    return pillow_image

# Initialize the object detector
ft_model_path = "runs/detect/train10/weights/best.pt"

ft_model = YOLO(ft_model_path).to('cpu')

# Queue for frames and bounding boxes
frame_queue = queue.Queue()
bounding_boxes_queue = queue.Queue()

def inference_thread():
    while True:
        pillow_image = frame_queue.get()
        if pillow_image is None:
            break
        results = ft_model(pillow_image)
        bounding_boxes = []
        for results in results:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().detach().numpy().astype(int)[0]
                #bounding_boxes.append([x2,x1, y2, y1])
                bounding_boxes.append([x1, y1, x2, y2])
        bounding_boxes_queue.put(bounding_boxes)

# Start the inference thread
thread = threading.Thread(target=inference_thread)
thread.start()

def stream_frames_with_bounding_boxes():
    c = 0
    bounding_boxes = []

    while True:
        frame = capture_dota2_window_opencv()
        
        # Convert to Pillow image
        pillow_image = opencv_to_pillow(frame)

        # Send frame for inference every 30 frames
        if c % 30 == 0:
            if not frame_queue.full():
                frame_queue.put(pillow_image)
        
        # Check if there are new bounding boxes
        if not bounding_boxes_queue.empty():
            bounding_boxes = bounding_boxes_queue.get()

        # Draw bounding boxes
        pillow_image = draw_bounding_boxes(pillow_image, bounding_boxes)
        
        # Convert back to OpenCV image
        frame_with_boxes = pillow_to_opencv(pillow_image)
        
        # Display the frame
        cv2.imshow("Dota 2 Stream with Bounding Boxes", frame_with_boxes)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        c += 1

    cv2.destroyAllWindows()
    # Stop the inference thread
    frame_queue.put(None)
    thread.join()


def query_cohere(bboxes):
    co = cohere.Client(
        api_key=COHERE_API_KEY
    )
    message = f"""
    You are an AI agent in a MOBA game. You are controlling a hero character.
    You are currently in a lane with enemy creeps. There are {len(bboxes)} number of creeps described by bounding boxes {bboxes}.

    Describe your next action by referencing a bounding box. You can either move to the bounding box and attack the creep or do nothing.
    If you choose to do nothing, return a negative one. If you choose to attack a creep, return the index of the bounding box you want to attack.
    
    Return only an integer."""
    chat = co.chat(
        message=message,
        model="command-r-plus"
    )
    return chat

def llm_play_dota():
    c = 0
    bounding_boxes = []

    while True:
        frame = capture_dota2_window_opencv()
        
        # Convert to Pillow image
        pillow_image = opencv_to_pillow(frame)

        # Send frame for inference every 30 frames
        if c % 30 == 0:
            if not frame_queue.full():
                frame_queue.put(pillow_image)
        
        # Check if there are new bounding boxes
        if not bounding_boxes_queue.empty():
            bounding_boxes = bounding_boxes_queue.get()

        # Query cohere with bounding box information
        if c % 30 == 0:
            action = query_cohere(bounding_boxes).text
            try:
                action = int(action)
            except:
                action = -1
            if action == -1:
                pass
            else:
                try:
                    move_mouse_to_center_of_bbox_and_attack(bounding_boxes[action])
                except IndexError:
                    pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        c += 1

    # Stop the inference thread
    frame_queue.put(None)
    thread.join()

# actually last hit
def move_mouse_to_center_of_bbox_and_attack(bbox):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2

    pyautogui.moveTo(x_center, y_center)

    # Right click
    pyautogui.click(button='right')    

llm_play_dota()