import cv2 
import numpy as np 
import time 
import os 
import mnsit_tracker as ht
import sys
import torch 
from torch.utils.data import Dataset , DataLoader , random_split 
import torch.nn as nn 
import matplotlib.pyplot as plt 
import pandas as pd 
import mediapipe as mp

device = "mps" if torch.backends.mps.is_available() else "cpu" 
model = nn.Sequential(
            nn.Linear(784,256), # hidden_layer_1 , 256 neurons
            nn.ReLU(), 
            nn.Linear(256,128), # hidden_layer_2 , 128 neurons
            nn.ReLU(), 
            nn.Linear(128,10)   
            ).to(device)
    
overlay_files = [
    "default.png",
    "red_1.png",
    "blue_2.png",
    "eraser_3.png",
    "pred_4.png"
]

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS   # PyInstaller temp folder
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

folderpath = resource_path("mnsit_canvas_images")

model_path = resource_path('/Users/sayar/Desktop/canvas/MNSIT CANVAS/model.pth')
model.load_state_dict(torch.load(model_path,map_location=device)) 
model.eval() 

overlay_list = []
for name in overlay_files:
    img = cv2.imread(os.path.join(folderpath, name))
    overlay_list.append(img)


header = overlay_list[0]  # bydefault image..

print("All Ok")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

detector = ht.HandDetector(DetCon=0.85)

thumb_start_time = None
THUMB_HOLD_TIME = 2  # seconds
current_header_index = 0

brush_thickness = 35
eraser_thickness = 250
xp,yp = 0,0

raw_headers = overlay_list
current_header_index = 0
header = None

tool_selected = False   # nothing selected initially
draw_color = None       # no color until user selects one


overlays = [
    {"img": overlay_list[0], "x": 150,  "y": 150, "w": 200, "h": 100, "id": 0 , "color": None}, #default 
    {"img": overlay_list[1], "x": 450,  "y": 150, "w": 200, "h": 100, "id": 1 , "color": (0,0,255)},     #red
    {"img": overlay_list[2], "x": 840,  "y": 150, "w": 200, "h": 100, "id": 2 , "color": (255,0,0)}, #blue  
    {"img": overlay_list[3], "x": 1200, "y": 150, "w": 200, "h": 100, "id": 3 , "color": "eraser"},     #eraser
    {"img": overlay_list[4], "x": 1600, "y": 150, "w": 200, "h": 100, "id": 4 , "color": (255,255,255)},#predictor   
]

def is_inside(x, y, box):
    return (box["x"] <= x <= box["x"] + box["w"] and 
            box["y"] <= y <= box["y"] + box["h"])
draw_color = (255,0,255) 
canvas_black = np.zeros((1080,1920,3),np.uint8)

last_save_time = 0
SAVE_COOLDOWN = 2   # seconds


while True: 
    success,frame = cap.read()
    frame = cv2.flip(frame,1)
    HEADER_HEIGHT = 160
    frame_width = frame.shape[1]

    header = cv2.resize(
        raw_headers[current_header_index],
        (frame_width, HEADER_HEIGHT))
    
    num_headers = len(overlay_list)
    section_width = frame.shape[1] // num_headers

    if not success: 
        print("Could Not Open Webcam..")
        break

    frame = detector.findhands(frame)
    lmlist = detector.findposition(frame,draw=False)
    if len(lmlist) != 0:         
        # tip of index,middle finger
        x1,y1 = lmlist[8][1:] 
        x2,y2 = lmlist[12][1:] 

        fingers = detector.fingers_up()

        index = current_header_index

        if fingers[1] and fingers[2]:
            print("Selection Mode")

            for item in overlays:
                if is_inside(x1,y1,item):

                    current_header_index = item["id"]

                    if item["color"] is None:
                        tool_selected = False
                        draw_color = None

                        canvas_black[:] = 0

                        if 'predicted_digit' in locals():
                            del predicted_digit

                        print("Canvas cleared → fresh start")



                    elif item["color"] == "eraser":
                        tool_selected = True
                        draw_color = "eraser"
                        print("Eraser selected")

                    elif item["id"] == 4:
                        tool_selected = False   # leave behaviour for later
                        print("Predict button selected")

                        if 'mnist_ready' in locals():

                            #mnist_ready = cv2.GaussianBlur(mnist_ready, (3,3), 0)
                            digit_tensor = torch.tensor(mnist_ready, dtype=torch.float32)
                            digit_tensor = digit_tensor.view(1,784).to(device)


                            with torch.no_grad():
                                output = model(digit_tensor)
                                probs = torch.softmax(output, dim=1)

                                pred = torch.argmax(output, dim=1).item()
                                confidence = probs[0, pred].item()

                                predicted_digit = pred
                                print("Predicted:", pred)
                                print("Confidence:", confidence)

                    else:
                        tool_selected = True
                        draw_color = item["color"]
                        print("Color selected:", draw_color)
                    break

            cv2.rectangle(frame,(x1,y1-25),(x2,y2+25),
                          draw_color if isinstance(draw_color,tuple) else (255,255,255),cv2.FILLED)

        if fingers[1] and not fingers[2]: 
            cv2.circle(frame,(x1,y1),15,(255,0,255),cv2.FILLED)
            print("Drawing Mode") 
            if xp == 0 and yp == 0: 
                xp,yp = x1,y1
                continue

              # ERASER
            if draw_color == "eraser":
                cv2.circle(frame, (x1, y1), eraser_thickness//2, (0,0,0), 3)
                cv2.line(canvas_black,(xp,yp),(x1,y1),(0,0,0),eraser_thickness)

            # NORMAL DRAW
            else:
                cv2.line(canvas_black,(xp,yp),(x1,y1),draw_color,brush_thickness)

            xp,yp = x1,y1
        
        else:
            xp,yp = 0,0

        cv2.putText(frame, f"x={x1}", (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.putText(frame, f"y={y1}", (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
                # Thumb-only detection
        if fingers[0] and not any(fingers[1:]):
            if thumb_start_time is None:
                thumb_start_time = time.time()
                print("Thumb detected... starting timer")

            elapsed = time.time() - thumb_start_time
            if elapsed >= THUMB_HOLD_TIME:
                print("Thumb held for 2 seconds. Shutting down.")
                break
        else:
            thumb_start_time = None  # reset if gesture breaks
    
    # ---------- DISPLAY PIPELINE (webcam + drawing) ----------
    frame_display = frame.copy()

    canvas_gray = cv2.cvtColor(canvas_black, cv2.COLOR_BGR2GRAY)
    _, canvas_mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)

    canvas_mask_inv = cv2.bitwise_not(canvas_mask)

    frame_bg = cv2.bitwise_and(frame_display, frame_display, mask=canvas_mask_inv)
    canvas_fg = cv2.bitwise_and(canvas_black, canvas_black, mask=canvas_mask)

    frame_display = cv2.add(frame_bg, canvas_fg)

    frame_display[0:HEADER_HEIGHT, :] = header
    
    if 'predicted_digit' in locals():
        cv2.putText(frame_display,
                f"Prediction: {predicted_digit}",(50,120),
                cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),4)
    cv2.imshow("Webcam", frame_display)

    # ---------- MNIST PIPELINE (background processing) ----------
    mnist_source = canvas_black[HEADER_HEIGHT:, :]
    mnist_gray = cv2.cvtColor(mnist_source, cv2.COLOR_BGR2GRAY)

    # make strokes white on black
    mnist_gray = cv2.GaussianBlur(mnist_gray, (5,5), 0)
    _, mnist_bin = cv2.threshold(mnist_gray, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mnist_bin) 

    if coords is not None:

        x,y,w,h = cv2.boundingRect(coords)
        digit = mnist_bin[y:y+h, x:x+w]

        digit = cv2.resize(digit, (20,20))

        mnist_ready = np.zeros((28,28), dtype=np.uint8)

        x_offset = (28 - 20)//2
        y_offset = (28 - 20)//2
        mnist_ready[y_offset:y_offset+20, x_offset:x_offset+20] = digit

        # normalize for model
        mnist_ready = mnist_ready / 255.0
        mnist_ready = (mnist_ready - 0.1307) / 0.3081

    if cv2.waitKey(1) & 0xff == ord('q'): 
        print("Quitting....")
        break
cap.release()
cv2.destroyAllWindows()