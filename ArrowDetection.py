################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
########## Arrow detection related logics resides here.#########
################################################################

# Imports
import cv2
import math

from CommonDirPaths import image_detected_dir_path, image_file_dir_path
from ConsoleOutMethods import displayArrows
from CustomClasses import Point, LineObject, Head, ArrowObject, Tail


def getArrowsFromImage(image_file_name):
    image_file_path = image_detected_dir_path + "Cut_" +image_file_name
    im = cv2.imread(image_file_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)

    # cv2.imshow("Thresh", thresh)
    cv2.imwrite(image_detected_dir_path + "Arrows_" + image_file_name, thresh)

    # On any error !!! Change a, from this line -> contours, hierarchy = cv2.findContours(thresh, 1, 2)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    arrows = findArrows(contours)
    displayArrows(arrows)
    # op_image = cv2.imread(image_file_dir_path + image_file_name)
    # op_image_arrowed = drawArrows(op_image, arrows)
    # cv2.imwrite(image_detected_dir_path + "Diag_Arrow_" + image_file_name, op_image_arrowed)
    return arrows


def findArrows(contours):
    arrows = []
    heads, lines = findHeadsAndLines(contours)

    for head in heads:
        nearest_line = findTheNearestLine(head, lines)
        tail_point = findTheTailPoint(head, nearest_line)
        arrows.append(ArrowObject(head, Tail(nearest_line.cnt, tail_point), nearest_line))

    return arrows


def findHeadsAndLines(contours):
    heads = []
    lines = []
    for c in contours:
        M = cv2.moments(c)
        # Removing noise data from the detected contours
        area = cv2.contourArea(c)
        if M["m00"] != 0 and area > 10:

            # Head detection. Contours with area less than 90 will be a head.
            if area < 90:
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                heads.append(Head(c, Point([cX, cY])))

            # Saving the line information for future processing.
            else:
                ext_left = tuple(c[c[:, :, 0].argmin()][0])
                ext_right = tuple(c[c[:, :, 0].argmax()][0])
                ext_top = tuple(c[c[:, :, 1].argmin()][0])
                ext_bot = tuple(c[c[:, :, 1].argmax()][0])
                lef_point = Point([ext_left[0], ext_left[1]])
                rig_point = Point([ext_right[0], ext_right[1]])
                top_point = Point([ext_top[0], ext_top[1]])
                bot_point = Point([ext_bot[0], ext_bot[1]])
                lines.append(LineObject(c, lef_point, rig_point, top_point, bot_point))

    return heads, lines


def findTheNearestLine(head, lines):
    head_point = head.point
    smallest_distance = getDistance(head_point, lines[0].lef)
    nearest_line = lines[0]
    for line in lines:
        dist_lef = getDistance(head_point, line.lef)
        dist_rig = getDistance(head_point, line.rig)
        dist_top = getDistance(head_point, line.top)
        dist_bot = getDistance(head_point, line.bot)

        if dist_lef < smallest_distance:
            smallest_distance = dist_lef
            nearest_line = line
        if dist_rig < smallest_distance:
            smallest_distance = dist_rig
            nearest_line = line
        if dist_top < smallest_distance:
            smallest_distance = dist_top
            nearest_line = line
        if dist_bot < smallest_distance:
            smallest_distance = dist_bot
            nearest_line = line
    return nearest_line


def findTheTailPoint(head, line):
    head_point = head.point
    largest_distance = getDistance(head_point, line.lef)
    tail_point = line.lef

    dist_lef = getDistance(head_point, line.lef)
    dist_rig = getDistance(head_point, line.rig)
    dist_top = getDistance(head_point, line.top)
    dist_bot = getDistance(head_point, line.bot)

    if dist_lef > largest_distance:
        largest_distance = dist_lef
        tail_point = line.lef
    if dist_rig > largest_distance:
        largest_distance = dist_rig
        tail_point = line.rig
    if dist_top > largest_distance:
        largest_distance = dist_top
        tail_point = line.top
    if dist_bot > largest_distance:
        tail_point = line.bot

    return tail_point


def drawArrows(im, arrows):
    cnt = 1
    for arrow in arrows:
        cv2.circle(im, (arrow.head.point.coords[0], arrow.head.point.coords[1]), 4, (0, 0, 255), -1)
        cv2.putText(im, 'H-' + str(cnt), (arrow.head.point.coords[0] - 3, arrow.head.point.coords[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.circle(im, (arrow.tail.point.coords[0], arrow.tail.point.coords[1]), 4, (0, 255, 0), -1)
        cv2.putText(im, 'T-' + str(cnt), (arrow.tail.point.coords[0] - 3, arrow.tail.point.coords[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cnt += 1

    cv2.imshow("Image", im)
    cv2.waitKey(0)
    return im


def evaluateArrowClusterRelation(arrows, clusters):
    arrow_cluster_map = []
    for arrow in arrows:
        head_cluster, tail_cluster = findHeadsAndTailClusters(arrow, clusters)
        arrow_cluster_map.append((arrow, head_cluster, tail_cluster))
    return arrow_cluster_map


def findHeadsAndTailClusters(arrow, clusters):
    head_point = arrow.head.point
    tail_point = arrow.tail.point
    head_cluster = findNearestClusterFromPoint(head_point, clusters)
    tail_cluster = findNearestClusterFromPoint(tail_point, clusters)
    return head_cluster, tail_cluster


def findNearestClusterFromPoint(point, clusters):
    smallest_distance = getDistance(point, clusters[0].centroid)
    nearest_cluster = clusters[0]
    for cluster in clusters:
        distance = getDistance(point, cluster.centroid)
        if distance < smallest_distance:
            smallest_distance = distance
            nearest_cluster = cluster

    return nearest_cluster


def getDistance(a, b):
    """
        Euclidean distance between two n-dimensional points.
        Note: This can be very slow and does not scale well
        """
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")

    ret = reduce(lambda x, y: x + pow((a.coords[y] - b.coords[y]), 2), range(a.n), 0.0)
    return math.sqrt(ret)
