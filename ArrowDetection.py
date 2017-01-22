import cv2
import imutils
import math
from PIL import Image

from CommonMethods import getDistance, imageDetectedDirPath, imageFileDirPath
from ConsoleOutMethods import displayArrows
from CustomClasses import Point, LineObject, Head, ArrowObject, Tail


def arrowEvaluation():
    imageFileName = imageDetectedDirPath + "Cut_proj_diag1_Mixed.jpg"
    im = cv2.imread(imageFileName)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)

    cv2.imshow("Thresh", thresh)

    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    arrows = findArrows(contours)
    displayArrows(arrows)
    opImage = cv2.imread(imageFileDirPath + "proj_diag1_Mixed.jpg")
    drawArrows(opImage, arrows)


def findArrows(contours):
    arrows = []
    heads , lines = findHeadsAndLines(contours)

    for head in heads:
        nearestLine = findTheNearestLine(head, lines)
        tailPoint = findTheTailPoint(head, nearestLine)
        arrows.append(ArrowObject(head, Tail(nearestLine.cnt, tailPoint), nearestLine))

    return arrows

def findHeadsAndLines(contours):
    heads = []
    lines = []
    for c in contours:
        M = cv2.moments(c)
        # Removing noise data from the detected contours
        area = cv2.contourArea(c)
        if (M["m00"] != 0 and area > 10):

            # Head detection. Contours with area less than 90 will be a head.
            if area < 90 :
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                heads.append(Head(c, Point([cX, cY])))

            # Saving the line information for future processing.
            else:
                extLeft = tuple(c[c[:, :, 0].argmin()][0])
                extRight = tuple(c[c[:, :, 0].argmax()][0])
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                lefPoint = Point([extLeft[0],extLeft[1]])
                rigPoint = Point([extRight[0],extRight[1]])
                topPoint = Point([extTop[0], extTop[1]])
                botPoint = Point([extBot[0],extBot[1]])
                lines.append(LineObject(c, lefPoint, rigPoint, topPoint, botPoint))

    return heads, lines

def findTheNearestLine(head, lines):
    headPoint = head.point
    smallest_distance = getDistance(headPoint, lines[0].lef)
    nearestLine = lines[0]
    for line in lines:
        distLef = getDistance(headPoint, line.lef)
        distRig = getDistance(headPoint, line.rig)
        distTop = getDistance(headPoint, line.top)
        distBot = getDistance(headPoint, line.bot)

        if distLef < smallest_distance:
            smallest_distance = distLef
            nearestLine = line
        if distRig < smallest_distance:
            smallest_distance = distRig
            nearestLine = line
        if distTop < smallest_distance:
            smallest_distance = distTop
            nearestLine = line
        if distBot < smallest_distance:
            smallest_distance = distBot
            nearestLine = line
    return nearestLine


def findTheTailPoint(head, line):
    headPoint = head.point
    largest_distance = getDistance(headPoint, line.lef)
    tailPoint = line.lef

    distLef = getDistance(headPoint, line.lef)
    distRig = getDistance(headPoint, line.rig)
    distTop = getDistance(headPoint, line.top)
    distBot = getDistance(headPoint, line.bot)

    if distLef > largest_distance:
        largest_distance = distLef
        tailPoint = line.lef
    if distRig > largest_distance:
        largest_distance = distRig
        tailPoint = line.rig
    if distTop > largest_distance:
        largest_distance = distTop
        tailPoint = line.top
    if distBot > largest_distance:
        tailPoint = line.bot

    return tailPoint


def drawArrows(im, arrows):
    cnt = 1
    for arrow in arrows:
        cv2.circle(im, (arrow.head.point.coords[0], arrow.head.point.coords[1]), 4, (0, 0, 255), -1)
        cv2.putText(im, 'H-' + str(cnt), (arrow.head.point.coords[0]-3, arrow.head.point.coords[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.circle(im, (arrow.tail.point.coords[0],arrow.tail.point.coords[1]), 4, (0, 255, 0), -1)
        cv2.putText(im, 'T-' + str(cnt), (arrow.tail.point.coords[0]-3, arrow.tail.point.coords[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cnt += 1

    cv2.imshow("Image", im)
    cv2.waitKey(0)

