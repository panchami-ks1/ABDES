################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
######## Diagram Evaluation related logics defined here.########
################################################################

# Imports
from dill import dill
from CommonMethods import processImage, generateAllContourPointsForClustering, image_file_dir_path, data_save_dir_path
from ConsoleOutMethods import displayClassificationResult
from KmeansClassification import kmeansClassification


def diagramEvaluation(image_name_for_evaluation):
    images = [processImage(image_file_dir_path, image_name_for_evaluation)]
    with open(data_save_dir_path + 'trained_data.pkl', 'rb') as f:
        data = dill.load(f)

    points = generateAllContourPointsForClustering(images)

    point_cluster_map = kmeansClassification(data.clusters, points)

    score = evaluateAnswerPoints(points, point_cluster_map)
    print "Score : ", score
    displayClassificationResult(point_cluster_map)


def evaluateAnswerPoints(points, point_cluster_map):
    # Checking whether the answer diagram and trained data has equal no of text blocks/regions or not.
    check = len(points) - len(point_cluster_map)
    if check == 0:
        score = evaluateAnswerPointsEqual(point_cluster_map)
    elif check < 0:
        score = evaluateAnswerPointsLesser(point_cluster_map)
    else:
        score = evaluateAnswerPointsGreater(point_cluster_map)

    return score


# To handle scenario with question diagrams having equal no of text regions/blocks.
def evaluateAnswerPointsEqual(point_cluster_map):
    print "Handling equal point - cluster scenario."
    count = 0
    for (point, cluster) in point_cluster_map:
        if cluster.points[0].text == point.text:
            count += 1

    return count


# To handle error scenario with question diagrams having lesser text regions/blocks.
def evaluateAnswerPointsLesser(point_cluster_map):
    print "Handling lesser point than cluster scenario."
    count = 0
    for (point, cluster) in point_cluster_map:
        if cluster:
            if cluster.points[0].text == point.text:
                count += 1
        else:
            count -= 1

    return count


# To handle error scenario with question diagrams having more text regions/blocks.
def evaluateAnswerPointsGreater(point_cluster_map):
    print "Handling greater point than cluster scenario."
    count = 0
    for (point, cluster) in point_cluster_map:
        if cluster:
            if cluster.points[0].text == point.text:
                count += 1
        else:
            count -= 1

    return count
