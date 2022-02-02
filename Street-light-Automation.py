import cv2,urllib,sys,math
import numpy as np

# FUNCTIONS

# executes first part of the program. i.e to find the difference between two frames
def getDifferenceHulls(host):
    # capturing two reference frames
    imgFrame2 = camread(host)

    # making duplicates of the above frames
    imgFrame1Copy = imgFrame1.copy()
    imgFrame2Copy = imgFrame2.copy()

    # changing the colorspace to grayscale
    imgFrame1Copy = cv2.cvtColor(imgFrame1Copy, cv2.COLOR_BGR2GRAY)
    imgFrame2Copy = cv2.cvtColor(imgFrame2Copy, cv2.COLOR_BGR2GRAY)

    # applying gaussianblur
    imgFrame1Copy = cv2.GaussianBlur(imgFrame1Copy, (5, 5), 0)
    imgFrame2Copy = cv2.GaussianBlur(imgFrame2Copy, (5, 5), 0)

    # finding the difference of the two frames and thresholding the diff
    imgDifference = cv2.absdiff(imgFrame1Copy, imgFrame2Copy)
    _, imgThresh = cv2.threshold(imgDifference, 30, 255, cv2.THRESH_BINARY)

    # cv2.imshow("imgThresh",imgThresh)

    # morphological operations: dilation and erosion
    kernel = np.ones((5, 5), np.uint8)
    imgThresh = cv2.dilate(imgThresh, kernel, iterations = 1)
    imgThresh = cv2.dilate(imgThresh, kernel, iterations = 1)
    imgThresh = cv2.erode(imgThresh, kernel, iterations = 1)


    # finding contours of the thresholded image
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # finding and drawing convex hulls
    hulls = []  # used to store hulls
    for cnt in contours:
        hulls.append(cv2.convexHull(cnt))

    return hulls, imgFrame2

# draws the rectangles on the motion detected object
def drawBlobInfoOnImage(blobs, imgFrame2Copy):
    for i in range(len(blobs)):
        if (blobs[i].blnStillBeingTracked == True):
            rect_corner1 = (blobs[i].currentBoundingRect[0], blobs[i].currentBoundingRect[1])
            rect_corner2 = (blobs[i].currentBoundingRect[0] + blobs[i].width, blobs[i].currentBoundingRect[1] + blobs[i].height)

            intFontFace = cv2.FONT_HERSHEY_SIMPLEX
            dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0
            intFontThickness = int(round(dblFontScale * 1.0))
            point = ((rect_corner1[0] + rect_corner2[0]) / 2, (rect_corner1[1] + rect_corner2[1]) / 2)
            cv2.putText(imgFrame2Copy, str(i), blobs[i].centerPositions[-1], intFontFace, dblFontScale, (0,255,0), intFontThickness);

            cv2.rectangle(imgFrame2Copy, rect_corner1, rect_corner2, (0, 0, 255))


# reads the data stream and decodes the image from it
def camread(host):
    stream = urllib.urlopen(host)
    bytes = ''
    while True:
        bytes += stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a !=-1 and b != -1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), -1)
            return i

# draws the contours on the image
def drawAndShowContours(imageSize, contours, strImageName):
    image = np.zeros(imageSize, dtype=np.uint8)
    cv2.drawContours(image, contours, -1, (255, 255, 255), -1)
    cv2.imshow(strImageName, image)

# draws the contours similar to the drawAndShowContours function
# but here the input provided is not the contours but object of class
# Blob
def drawAndShowBlobs(imageSize, blobs, strImageName):
    image = np.zeros(imageSize, dtype=np.uint8)
    contours = []
    for blob in blobs:
        if blob.blnStillBeingTracked == True:
            contours.append(blob.currentContour)

    cv2.drawContours(image, contours, -1, (255,255,255), -1)
    cv2.imshow(strImageName, image)

# find the distance between two points p1 and p2
def distanceBetweenPoints(point1, point2):
    intX = abs(point1[0] - point2[0])
    intY = abs(point1[1] - point2[1])
    return math.sqrt(math.pow(intX, 2) + math.pow(intY, 2))

# matching algorithm to corelate two blob objects by matching it with the expected one
def matchCurrentFrameBlobsToExistingBlobs(existingBlobs, currentFrameBlobs):
    for existingBlob in existingBlobs:
        existingBlob.blnCurrentMatchFoundOrNewBlob = False
        existingBlob.predictNextPosition()

    for currentFrameBlob in currentFrameBlobs:
        intIndexOfLeastDistance = 0
        dblLeastDistance = 100000.0

        for i in range(len(existingBlobs)):
            if existingBlobs[i].blnStillBeingTracked == True:
                dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions[-1], existingBlobs[i].predictedNextPosition)
                # print dblDistance
                if dblDistance < dblLeastDistance:
                    dblLeastDistance = dblDistance
                    intIndexOfLeastDistance = i

        if dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 1.15:
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance)
        else:
            addNewBlob(currentFrameBlob, existingBlobs)


    for existingBlob in existingBlobs:
        if existingBlob.blnCurrentMatchFoundOrNewBlob == False:
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch += 1

        if existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5:
            existingBlob.blnStillBeingTracked = False

# adds the details of the matching blob to the existingBlob
def addBlobToExistingBlobs(currentFrameBlob,existingBlobs,i):
    # print 'found continuos blob'
    existingBlobs[i].currentContour = currentFrameBlob.currentContour
    existingBlobs[i].currentBoundingRect = currentFrameBlob.currentBoundingRect

    existingBlobs[i].centerPositions.append(currentFrameBlob.centerPositions[-1])

    existingBlobs[i].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize
    existingBlobs[i].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio

    existingBlobs[i].blnStillBeingTracked = True
    existingBlobs[i].blnCurrentMatchFoundOrNewBlob = True

# adds new blob to the list
def addNewBlob(currentFrameBlob, existingBlobs):
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = True
    existingBlobs.append(currentFrameBlob)


# CLASS
# class Blob consisting of variables and functions related to it
class Blob:
    # functions
    def printInfo(self):
        print 'area: ' + str(self.area) + ' Pos: ' + str(self.centerPositions)

    def __init__(self, _contour):

        self.centerPositions = []
        self.predictedNextPosition = [-1, -1]

        self.currentContour = _contour
        self.currentBoundingRect = cv2.boundingRect(self.currentContour)  #x,y,w,h
        x = (self.currentBoundingRect[0] + self.currentBoundingRect[0] + self.currentBoundingRect[2]) / 2
        y = (self.currentBoundingRect[1] + self.currentBoundingRect[1] + self.currentBoundingRect[3]) / 2
        currentCenter = (x, y)
        self.width = self.currentBoundingRect[2]
        self.height =  self.currentBoundingRect[3]
        self.area = self.currentBoundingRect[2] * self.currentBoundingRect[3]

        self.centerPositions.append(currentCenter)

        self.dblCurrentDiagonalSize = math.sqrt(math.pow(self.currentBoundingRect[2], 2) + math.pow(self.currentBoundingRect[3], 2));
        self.dblCurrentAspectRatio = float(self.currentBoundingRect[2]) / float(self.currentBoundingRect[3])

        self.blnStillBeingTracked = True
        self.blnCurrentMatchFoundOrNewBlob = True

        self.intNumOfConsecutiveFramesWithoutAMatch = 0

    def predictNextPosition(self):
        numPositions = len(self.centerPositions)
        if numPositions == 1:
            self.predictedNextPosition[0] = self.centerPositions[-1][0]
            self.predictedNextPosition[1] = self.centerPositions[-1][1]

        elif numPositions == 2:
            deltaX = self.centerPositions[1][0] - self.centerPositions[0][0]
            deltaY = self.centerPositions[1][1] - self.centerPositions[0][1]

            self.predictedNextPosition[0] = self.centerPositions[-1][0] + deltaX
            self.predictedNextPosition[1] = self.centerPositions[-1][1] + deltaY

        elif numPositions == 3:
            sumOfXChanges = ((self.centerPositions[2][0] - self.centerPositions[1][1]) * 2) + \
            ((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)

            deltaX = int(round(float(sumOfXChanges)/3.0))

            sumOfYChanges = ((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) + \
            ((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)

            deltaY = int(round(float(sumOfYChanges) / 3.0))

            self.predictedNextPosition[0] = self.centerPositions[-1][0] + deltaX
            self.predictedNextPosition[1] = self.centerPositions[-1][1] + deltaY

        elif numPositions == 4:
            sumOfXChanges = ((self.centerPositions[3][0] - self.centerPositions[2][0]) * 3) + \
            ((self.centerPositions[2][0] - self.centerPositions[1][0]) * 2) + \
            ((self.centerPositions[1][0] - self.centerPositions[0][0]) * 1)

            deltaX = int(round(float(sumOfXChanges) / 6.0))

            sumOfYChanges = ((self.centerPositions[3][1] - self.centerPositions[2][1]) * 3) + \
            ((self.centerPositions[2][1] - self.centerPositions[1][1]) * 2) + \
            ((self.centerPositions[1][1] - self.centerPositions[0][1]) * 1)

            deltaY = int(round(float(sumOfYChanges) / 6.0))

            self.predictedNextPosition[0] = self.centerPositions[-1][0] + deltaX;
            self.predictedNextPosition[1] = self.centerPositions[-1][1] + deltaY;

        elif numPositions >= 5:
            sumOfXChanges = ((self.centerPositions[numPositions - 1][0] - self.centerPositions[numPositions - 2][0]) * 4) + \
            ((self.centerPositions[numPositions - 2][0] - self.centerPositions[numPositions - 3][0]) * 3) + \
            ((self.centerPositions[numPositions - 3][0] - self.centerPositions[numPositions - 4][0]) * 2) + \
            ((self.centerPositions[numPositions - 4][0] - self.centerPositions[numPositions - 5][0]) * 1)

            deltaX = int(round(float(sumOfXChanges) / 10.0));

            sumOfYChanges = ((self.centerPositions[numPositions - 1][1] - self.centerPositions[numPositions - 2][1]) * 4) + \
            ((self.centerPositions[numPositions - 2][1] - self.centerPositions[numPositions - 3][1]) * 3) + \
            ((self.centerPositions[numPositions - 3][1] - self.centerPositions[numPositions - 4][1]) * 2) + \
            ((self.centerPositions[numPositions - 4][1] - self.centerPositions[numPositions - 5][1]) * 1)

            deltaY = int(round(float(sumOfYChanges) / 10.0))

            self.predictedNextPosition[0] = self.centerPositions[-1][0] + deltaX;
            self.predictedNextPosition[1] = self.centerPositions[-1][1] + deltaY;

        else:
            # should never get here
            pass


# MAIN CODE
cap = cv2.VideoCapture('video.avi')
light = cv2.imread('light.jpg')
ref_rows, ref_cols, _ = light.shape

# variables used within the infinite loop
blnFirstFrame = True
frameCount = 2                  # counts the number of frames captured
if cap.isOpened():
    _, imgFrame1 = cap.read()   # capturing the first reference frame
else:
    sys.exit()
blobs = []                      # captures all the new blobs found

while cap.isOpened():
    # obtaining convex hulls and newly captured image
    hulls, imgFrame2 = getDifferenceHulls(cap)

    # Blob validation
    currentFrameBlobs = []
    for hull in hulls:
        possibleBlob = Blob(hull)

        #conditions to approximate the blobs
        if (possibleBlob.area > 100 and \
        possibleBlob.dblCurrentAspectRatio >= 0.2 and \
        possibleBlob.dblCurrentAspectRatio <= 1.75 and \
        possibleBlob.width > 20 and \
        possibleBlob.height > 20 and \
        possibleBlob.dblCurrentDiagonalSize > 30.0 and \
        (cv2.contourArea(possibleBlob.currentContour) / float(possibleBlob.area)) > 0.40):

            currentFrameBlobs.append(possibleBlob)
            # possibleBlob.printInfo()
        del possibleBlob

    # replacing the frame1 with frame2, so that newly captured frame can be stored in frame2
    imgFrame1 = imgFrame2.copy()

    # mouse tracking
    cv2.setMouseCallback('output', detect_point)

    # displaying any movement in the output screen
    if len(currentFrameBlobs) > 0:
        drawAndShowBlobs(imgFrame2.shape, currentFrameBlobs, "imgCurrentFrameBlobs")
        drawBlobInfoOnImage(currentFrameBlobs,imgFrame2)

    # checks if the frame is the first frame of the video
    if blnFirstFrame == True:
        for currentFrameBlob in currentFrameBlobs:
            blobs.append(currentFrameBlob)
    else:
        matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs)

    # displays the blobs on the screen that are consistent or matched
    drawAndShowBlobs(imgFrame2.shape, blobs, "imgBlobs")


    cv2.ellipse(imgFrame2, (436, 381), (200, 100), 0, 0, 360, (0, 0, 255), 5)
    peopleCount = 0
    for currentFrameBlob in currentFrameBlobs:
        if currentFrameBlob.inside:
            peopleCount += 1
            cv2.line(imgFrame2, currentFrameBlob.currentCenter, (currentFrameBlob.currentCenter[0], currentFrameBlob.currentCenter[1] + currentFrameBlob.height / 2), [255, 0, 0], 2)
    # print peopleCount 

    if peopleCount > 0:
        imgFrame2[20:20+ref_rows, 380:380+ref_cols] = light

    cv2.imshow("output", imgFrame2)
    # flagging the further frames
    blnFirstFrame = False
    frameCount += 1             # increments the number of frames
    del currentFrameBlobs[:]    # clearing the currentFrameBlobs to capture newly formed blobs
    cv2.waitKey(85)

cap.release()
cv2.destroyAllWindows()