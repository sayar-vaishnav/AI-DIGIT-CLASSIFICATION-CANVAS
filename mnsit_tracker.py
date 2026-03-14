import cv2 
import mediapipe as mp
import time 

class HandDetector(): 
    def __init__(self,Mode=False,MaxHands=2,DetCon=0.5,TrackCon=0.5):
        self.Mode = Mode
        self.MaxHands = MaxHands
        self.DetCon = DetCon
        self.TrackCon = TrackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode = self.Mode,
            max_num_hands = self.MaxHands, 
            min_detection_confidence=self.DetCon,
            min_tracking_confidence=self.TrackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    def findhands(self,img,draw=True):      
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks: 
            for handsLM in self.results.multi_hand_landmarks:
                if draw: 
                    self.mpDraw.draw_landmarks(img, handsLM ,
                                               self.mp_hands.HAND_CONNECTIONS)
        return img
        
    def findposition(self,img,handno=0,draw=True):
        self.lmlist = []
        if self.results.multi_hand_landmarks: 
            my_hand = self.results.multi_hand_landmarks[handno]
            for id,lm in enumerate(my_hand.landmark):
                h,w,c = img.shape
                cx , cy = int(lm.x*w) , int(lm.y*h)
                self.lmlist.append([id,cx,cy])
                if draw: 
                     cv2.circle(img,(cx,cy), 10,(255,0,255),cv2.FILLED)
        return self.lmlist
    
    def fingers_up(self): 
        fingers = []
        #thumb
        if self.lmlist[self.tipIds[0]][1] < self.lmlist[self.tipIds[0]-1][1]: 
            fingers.append(1) 
        else: 
            fingers.append(0)
        # 4 fingers
        for id in range(1,5): 
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[2]-2][2]: 
                fingers.append(1) 
            else: 
                fingers.append(0)
        return fingers

        


def main(): 
    Ptime = 0
    Ctime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Could not read frame")
            break
        
        img = detector.findhands(img)
        lmlist = detector.findposition(img)
        if len(lmlist) != 0:
            print(lmlist[4])
        
        
        Ctime = time.time()
        fps = 1/(Ctime-Ptime)
        Ptime = Ctime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
        cv2.imshow("Webcam Feed", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            print("Quitting...")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": 
    main()