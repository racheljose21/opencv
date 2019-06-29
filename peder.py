from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
#imagePaths = list(paths.list_images(args["images"]))

capture = cv2.VideoCapture('videoplayback.mp4')
#capture = cv2.VideoCapture('D:\susmi\krkk')
#largest_r = 0

while capture.isOpened():
    # grab the current frame and initialize the status text
    grabbed, frame = capture.read()
    if not  grabbed:
        break

    if frame is not None:
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        orig = frame.copy()

	# detect people in the image
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
             padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
        #filename = imagePath[imagePath.rfind("/") + 1:]
        print("[INFO] : {} people".format(len(rects)))

	# show the output images
        cv2.imshow("People detected", frame)
#	cv2.imshow("After NMS", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()