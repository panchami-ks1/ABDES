
def showClusters(clusters):
    for i, cluster in enumerate(clusters):
        print "Cluster " + str( i +1) + " : Text : " + cluster.points[0].text
        for point in cluster.points:
            print "( " + str(point.coords) + " )"
