################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
##### Custom Classes used for the project are defined here.#####
################################################################

# Imports
import cv2


# ImageObject class is used for saving the complete details of an input image.
# It contains the following information's
# 1. Image Name for refference.
# 2. A List containing individual text regions detected from the image.
class ImageObject():
    def __init__(self, image, contourList):
        self.image = image
        self.contourList = contourList


# ContourObject class is used for saving the details of adetected text region/contour region.
# It contains the following information's
# 1. Detected Contour feature.
# 2. x, y co-ordinate details.
# 3. Centroid of the detected contour region.
class ContourObject():
    def __init__(self, contour, x, y):
        # type: (object, object, object) -> object
        self.contour = contour
        self.x = x
        self.y = y
        self.cX, self.cY = self.findCentroid(contour, x, y)
    # Method to find the centroid of the contour object.
    def findCentroid(self, contour, x, y):
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
