################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
##### Custom Classes used for the project are defined here.#####
################################################################

# Imports
from PIL import Image
import dill
import cv2
import time
import numpy as np

import pytesser
from CustomClasses import ContourObject, ImageObject, Point, TrainedData
from Kmeans import kmeans
from KmeansClassification import kmeansClassification
import matplotlib.pyplot as plt


# Method will create a ContourObject and add it to the contourList given in the argument.
# arguments : (contourList, cnt, x, y)
# return : void
def addCountourToList(contourList, cnt, x, y, text):
    if len(contourList) == 0 and text != "":
        contourList.append(ContourObject(cnt, x, y, text))
    else:
        flag = True
        for contourVar in contourList:
            tempX = contourVar.x - x
            tempY = contourVar.y - y
            if tempY < 6 and tempX < 6:
                print "Point removed" + str(x) + str(y)
                flag = False
            if x == 1 and y == 1:
                flag = False
                print "Point removed" + str(x) + str(y)
        if flag and text != "":
            contourObject = ContourObject(cnt, x, y, text)
            if contourObject.cX != 0 and contourObject.cY != 0:
                contourList.append(contourObject)
                print x, y, text
    pass


# detected contour region details.
# arguments : (imageFileName)
# return : ImageObject
def processImage(imageFileName):
    im = cv2.imread(imageFileName)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray, 127, 255, 0)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    deleteRegions(im,contours)
    idx = 0
    lis=[]
    contourList = []
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            cv2.rectangle(im, (x, y), (x + w, y + h), (200, 0, 0), 2)
            roi = im[y:y + h, x:x + w]
            fileName = 'images/tmp/' + str(idx) + '.jpg'
            cv2.imwrite(fileName, roi)
            #time.sleep(0.5)
            img = Image.open(fileName)
            #plt.plot(roi)
            text = pytesser.image_to_string(img).strip()
            print text
            lis.append(text)
            addCountourToList(contourList, cnt, x, y, text)


    #print lis
    return ImageObject(im, contourList)

def generateInitialClusterPoints(images):
    initContourPoints = []
    initContoursList = images[0].contourList
    #print len(images[0].contourList)
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
def showClusters(clusters):
    for i, cluster in enumerate(clusters):
        print "Cluster " + str(i+1) + " : Text : " + cluster.points[0].text
        for point in cluster.points:
            print "( " + str(point.coords) + " )"

# The main method from where the all project execution begins.
# arguments : void
# return : void
def saveTrainingData(images, clusters):
    trainedData = TrainedData(images,clusters)

    with open('trained_data.pkl', 'wb') as f:
        dill.dump(trainedData, f)
    pass

def diagramEvaluation():
        images = []
        images.append(processImage('images/diagram.jpg'))
        with open('trained_data.pkl', 'rb') as f:
            data = dill.load(f)

        points = generateAllContourPointsForClustering(images)

        clusters = kmeansClassification(data.clusters, points)

        evaluateAnswerPoints(points, clusters)

        showClusters(clusters)



def evaluateAnswerPoints(points, clusters):
    # Checking whether the answer diagram and trained data has equal no of text blocks/regions or not.
    check = len(points) - len(clusters)

    if check == 0:
        evaluateAnswerPointsEqual(points, clusters)
    elif check < 0:
        evaluateAnswerPointsLesser(points, clusters)
    else:
        evaluateAnswerPointsGreater(points, clusters)

# To handle scenario with question diagrams having equal no of text regions/blocks.
def evaluateAnswerPointsEqual(points, clusters):
    print "Handling equal point - cluster scenario."
    count = 0
    for p in points:
        cluster = findCluster(p, clusters)
        if cluster.points[1].text == p.text:
            count += 1
    print "Score : ", count
    pass


# To handle error scenario with question diagrams having lesser text regions/blocks.
def evaluateAnswerPointsLesser(points, clusters):
    print "Handling lesser point than cluster scenario."
    count = 0
    for p in points:
        cluster = findCluster(p, clusters)
        if cluster:
           if cluster.points[1].text == p.text:
            count += 1
        else:
            count -=1
    print "Score : ", count
    pass

# To handle error scenario with question diagrams having more text regions/blocks.
def evaluateAnswerPointsGreater(points, clusters):
    print "Handling greater point than cluster scenario."
    count = 0
    for p in points:
        cluster = findCluster(p, clusters)
        if cluster:
            if cluster.points[1].text == p.text:
                count += 1
        else:
            count -= 1

    print "Score : ", count
    pass


def findCluster(p,clusters):
    for i, cluster in enumerate(clusters):
         if p in cluster.points:
              return cluster

def deleteRegions(imageName,contour):

    mask = np.ones(imageName.shape[:2], dtype="uint8") * 255
    cv2.imwrite('images/deleted/mask.jpg', mask)
    for c in contour:
        # if the contour is bad, draw it on the mask
        if is_contour_bad(c):
            cv2.drawContours(mask, [c], -1, 0, -1)

    # remove the contours from the image and show the resulting images
    image1 = cv2.bitwise_not(imageName, mask=mask)

    cv2.imwrite('images/deleted/edge.jpg', mask)
    cv2.imwrite('images/deleted/original.jpg', image1)


def is_contour_bad(c):
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # the contour is 'bad' if it is not a rectangle
    return  len(approx) == 4

def main():
    images = []
    images.append(processImage('images/diag1.jpg'))
    images.append(processImage('images/diag1.jpg'))



    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['green', 'blue', 'pink', 'gold', 'red', 'yellow', 'black']

    #Populate initial points for clustering
    initContourPoints = generateInitialClusterPoints(images)
    #print len(initContourPoints)


    # Generate some points
    points = generateAllContourPointsForClustering(images)
    # When do we say the optimization has 'converged' and stop updating clusters
    opt_cutoff = 0.5

    # Cluster those data!
    clusters = kmeans(initContourPoints, points, opt_cutoff)
    showClusters(clusters)

    saveTrainingData(images, clusters)

    # Draw the cluster points
    X = []
    Y = []

    Xc = []
    Yc = []
    plt.axis([0, 800, 0, 800])
    for i, c in enumerate(clusters):
        X = []
        Y = []
        Xc = [c.centroid.coords[0]]
        Yc = [c.centroid.coords[1]]
        for p in c.points:
            X.append(p.coords[0])
            Y.append(p.coords[1])
        plt.plot(X, Y, 'w', markerfacecolor=colors[i], marker='.', markersize=10)
        plt.plot(Xc, Yc, 'o', markerfacecolor=colors[i], marker='*', markeredgecolor='k', markersize=10)
    plt.show()

