################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
##### Custom Classes used for the project are defined here.#####
################################################################


# Imports
import cv2
from CustomClasses import ContourObject, ImageObject
import matplotlib.pyplot as plt


# Method will create a ContourObject and add it to the contourList given in the argument.
# arguments : (contourList, cnt, x, y)
# return : void
def addCountourToList(contourList, cnt, x, y):
    if len(contourList) == 0:
        contourList.append(ContourObject(cnt, x, y))
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
        if flag:
            contourList.append(ContourObject(cnt, x, y))
    pass


# Method will processing the input image by identifying the text blocks present in it and returns a ImageObject with
# detected contour region details.
# arguments : (imageFileName)
# return : ImageObject
def processImage(imageFileName):
    im = cv2.imread(imageFileName)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    a, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    idx = 0
    contourList = []
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            addCountourToList(contourList, cnt, x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (200, 0, 0), 2)
            roi = im[y:y + h, x:x + w]
            cv2.imwrite('tmp2/' + str(idx) + '.jpg', roi)
            print x, y
    return ImageObject(im, contourList)


# The main method from where the all project execution begins.
# arguments : void
# return : void
def main():
    images = []
    images.append(processImage('diag.jpg'))

    images.append(processImage('sdiag.jpg'))


    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['green', 'blue', 'pink', 'gold', 'red', 'yellow', 'black']

    plt.axis([0, 800, 0, 800])
    for image in images:
        for i, countour in enumerate(image.contourList):
            X =(countour.x)
            Y = (countour.x)
            Xc =(countour.cX)
            Yc=(countour.cY)
            plt.plot(Xc, Yc, 'o', markerfacecolor=colors[i], marker='*', markeredgecolor='k', markersize=10)

    plt.show()

