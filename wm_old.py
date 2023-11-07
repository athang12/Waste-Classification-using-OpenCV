import os

import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)
classifier = Classifier("C:\\Users\\athan\\Downloads\\Resources\\Resources\\Model_old\\keras_model.h5","C:\\Users\\athan\\Downloads\\Resources\\Resources\\Model_old\\labels.txt")
imgArrow = cv2.imread("C:\\Users\\athan\\Downloads\\Resources\\Resources\\arrow.png", cv2.IMREAD_UNCHANGED)
classIDBin = 0
# Import all the waste images
imgWasteList = []
pathFolderWaste = "C:\\Users\\athan\\Downloads\\Resources\\Resources\\Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

# Import all the waste images
imgBinsList = []
pathFolderBins = "C:\\Users\\athan\\Downloads\\Resources\\Resources\\Bins"
pathList = os.listdir(pathFolderBins)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))

# 0 = Recyclable
# 1 = Hazardous
# 2 = Food
# 3 = Residual

classDic = {0: None,
            1: 0,
            2: 0,
            3: 3,
            4: 3,
            5: 1,
            6: 1,
            7: 2,
            8: 2}

while True:
    ret,img = cap.read()
    imgResize = cv2.resize(img, (100,50))

    imgBackground = cv2.imread("C:\\Users\\athan\\Downloads\\Resources\\Resources\\background.png")

    predection = classifier.getPrediction(img)

    classID = predection[1]
    print(classID)
    if classID != 0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID - 1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))

        classIDBin = classDic[classID]

    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (895, 374))

    imgBackground[148:148 + 50, 159:159 + 100] = imgResize
    # Displays
    # cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)
