################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
##### Custom Classes used for the project are defined here.#####
################################################################

# Imports
import math
from PIL import Image
import cv2
from PIL import ImageFilter
import pytesser
from CustomClasses import ContourObject, ImageObject, Point

imageFileDirPath = "images/diagrams/"
imageTempDirPath = "images/temp/"
imageDetectedDirPath = "images/detected/"
dataSaveDirPath = "data/"

# Detected contour region details.
# arguments : (imageFileName)
# return : ImageObject


def processImage(path, inputFileName):
    imageFileName = path + inputFileName
    im = cv2.imread(imageFileName)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = Image.open(imageFileName)
    #ret, thresh = cv2.threshold(gray, 127, 255, 0)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    idx = 0
    contourList = []
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        peri = cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        if len(approx) == 4:
            # cv2.rectangle(im, (x, y), (x + w, y + h), (200, 0, 0), 2)
            roi = im[y:y + h, x:x + w]
            fileName = imageTempDirPath + str(idx) + '.jpg'
            cv2.imwrite(fileName, roi)

            img = Image.open(fileName)

            text = pytesser.image_to_string(img).strip()
            #print text
            if (addCountourToList(contourList, cnt, x, y, text)):
                fileName = imageDetectedDirPath + str(idx) + '.jpg'
                cv2.imwrite(fileName, roi)
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
                image = removeImageContent(image, [x-3, y-3, (x + w + 3), (y + h + 3)])
                print "Text :", text, "$$"
    image.save(imageDetectedDirPath + "Cut_" + inputFileName)
    imageObject = ImageObject(im, contourList)
    imageObject.image_cut = image
    return ImageObject(im, contourList)


# Method will create a ContourObject and add it to the contourList given in the argument.
# arguments : (contourList, cnt, x, y)
# return : void
def addCountourToList(contourList, cnt, x, y, text):
    flag = False
    if len(contourList) == 0 and text != "":
        contourList.append(ContourObject(cnt, x, y, text))
        flag = True
    else:
        flag = True
        for contourVar in contourList:
            tempX = contourVar.x - x
            tempY = contourVar.y - y
            if tempY < 5 and tempX < 5:
                print "Point removed" + str(x) + str(y)
                flag = False
            if x == 1 and y == 1:
                flag = False
                print "Point removed" + str(x) + str(y)
        if flag and (text != "" or len(text) != 0):
            contourObject = ContourObject(cnt, x, y, text)
            if contourObject.cX != 0 and contourObject.cY != 0:
                contourList.append(contourObject)
                print x, y, text
            else:
                flag = False
        else:
            flag = False

    return flag


def removeImageContent(image, positions):
    image_crop_part = image.crop(positions)
    for i in range(100):  # You can blur many times
        image_crop_part = image_crop_part.filter(ImageFilter.BLUR)
    image.paste(image_crop_part, positions)
    return image


def generateInitialClusterPoints(images):
    initContourPoints = []
    initContoursList = images[0].contourList
    for contour in initContoursList:
        initContourPoints.append(Point([contour.cX, contour.cY], contour.text))
    return initContourPoints


def generateAllContourPointsForClustering(images):
    contourPoints = []
    for image in images:
        contourList = image.contourList
        for contour in contourList:
            contourPoints.append(Point([contour.cX, contour.cY], contour.text))
    return contourPoints


def getDistance(a, b):
    '''
        Euclidean distance between two n-dimensional points.
        Note: This can be very slow and does not scale well
        '''
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")

    ret = reduce(lambda x, y: x + pow((a.coords[y] - b.coords[y]), 2), range(a.n), 0.0)
    return math.sqrt(ret)