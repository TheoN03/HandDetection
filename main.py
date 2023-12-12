import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

# Model initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_tracking_confidence=0.75,
    max_num_hands=2)

# Video capturing from webcam
cap = cv2.VideoCapture(0)

# Read and flip the video by frame
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Conversion to RBG
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processing the RGB image
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        if len(results.multi_handedness) == 2:
            cv2.putText(img, "Both Hands", (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9, (0, 255, 0), 2)
        else:
            for i in results.multi_handedness:
                label = MessageToDict(i)["classification"][0]["label"]

                if label == "Left":
                    cv2.putText(img, label + " Hand",
                                (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)

                if label == "Right":
                    cv2.putText(img, label + " Hand", (460, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)

    # Display Video and the pressing "q" command for destroying the window
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
