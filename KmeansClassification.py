import math

from CommonMethods import getDistance


def kmeansClassification(clusters, points):
    loopCounter = 0
    lists = [c.points for c in clusters]
    clusterCount = len(clusters)

    # For every point in the dataset ...
    for p in points:
        # Get the distance between that point and the centroid of the first
        # cluster.
        smallest_distance = getDistance(p, clusters[0].centroid)

        # Set the cluster this point belongs to
        clusterIndex = 0

        # For the remainder of the clusters ...
        for i in range(clusterCount - 1):
            # calculate the distance of that point to each other cluster's
            # centroid.
            distance = getDistance(p, clusters[i + 1].centroid)

            if distance < smallest_distance:
                smallest_distance = distance
                clusterIndex = i + 1
        threshold=50
        if smallest_distance>50:
            print "Removed"
        else:
            lists[clusterIndex].append(p)
    # Set our biggest_shift to zero for this iteration
    biggest_shift = 0.0

    # As many times as there are clusters ...
    for i in range(clusterCount):
        # Calculate how far the centroid moved in this iteration
        shift = clusters[i].update(lists[i])
    return clusters