################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
####### Methods for printing data in console defined here.######
################################################################


def showClusters(clusters):
    for i, cluster in enumerate(clusters):
        print "Cluster " + str(i + 1) + " : Text : " + cluster.points[0].text
        for point in cluster.points:
            print "( " + str(point.coords) + " )"


def displayArrows(arrows):
    cnt = 1
    for arrow in arrows:
        print "Arrow", cnt, "Head : ", arrow.head.point, "Tail : ", arrow.tail.point
        cnt += 1
