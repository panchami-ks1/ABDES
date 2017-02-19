################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
######## Diagram Evaluation related logics defined here.########
################################################################

# Imports
from dill import dill
from ArrowDetection import evaluateArrowClusterRelation
from CommonDirPaths import image_file_dir_path, data_save_dir_path
from CommonMethods import processImage, generateAllContourPointsForClustering, generateInitialClusterPoints
from ConsoleOutMethods import displayClassificationResult
from Kmeans import kmeans
from KmeansClassification import kmeansClassification
from SVMImplementation import predictFromTrainData, predictScoreForDiagram


def isBlocksMatched(trained_block, input_block):
    if trained_block.text == input_block.text:
        return True
    else:
        return False


def isArrowsMatched(trained_arrow, input_arrow):
    flag = True
    (trained_arr, trained_head, trained_tail) = trained_arrow
    (input_arr, input_head, input_tail) = input_arrow
    if trained_head.centroid.text != input_head.centroid.text:
        flag = False

    if trained_tail.centroid.text != input_tail.centroid.text:
        flag = False

    return flag


def blockEvaluationNew(trained_blocks, input_blocks, flag):
    trained_visited_list = ["N"] * len(trained_blocks)
    input_visited_list = ["N"] * len(input_blocks)

    for i, trained_block in enumerate(trained_blocks):
        for j, input_block in enumerate(input_blocks):
            if isBlocksMatched(trained_block, input_block) and input_visited_list[j] == "N":
                trained_visited_list[i] = "Y"
                input_visited_list[j] = "Y"
                break

    print trained_visited_list
    print input_visited_list

    if flag:
        return 0, trained_visited_list

    else:
        if len(trained_blocks) > len(input_blocks):
            return 0, trained_visited_list

        else:
            return len(input_blocks) - len(trained_blocks), trained_visited_list

    pass


def arrowEvaluationNew(trained_arrows, input_arrows, flag):
    trained_visited_list = ["N"] * len(trained_arrows)
    input_visited_list = ["N"] * len(input_arrows)

    for i, trained_arrow in enumerate(trained_arrows):
        for j, input_arrow in enumerate(input_arrows):
            if isArrowsMatched(trained_arrow, input_arrow) and input_visited_list[j] == "N":
                trained_visited_list[i] = "Y"
                input_visited_list[j] = "Y"
                break

    print trained_visited_list
    print input_visited_list
    if flag:
        return 0, trained_visited_list

    else:
        if len(trained_arrows) > len(input_arrows):
            return 0, trained_visited_list

        else:
            return len(input_arrows) - len(trained_arrows), trained_visited_list

    pass


def diagramEvaluationNew(image_name_for_evaluation):
    print image_name_for_evaluation
    images = [processImage(image_file_dir_path, image_name_for_evaluation)]

    # Populate initial points for clustering
    init_contour_points = generateInitialClusterPoints(images)

    # Generate some points
    points = generateAllContourPointsForClustering(images)
    # When do we say the optimization has 'converged' and stop updating clusters
    opt_cutoff = 0.5

    # Cluster those data!
    clusters = kmeans(init_contour_points, points, opt_cutoff)

    arrow_cluster_map = evaluateArrowClusterRelation(images[0].arrows, clusters)

    with open(data_save_dir_path + 'trained_data.pkl', 'rb') as f:
        data = dill.load(f)

    trained_blocks = data.images[0].contour_list
    trained_arrows = data.arrow_cluster_map
    trained_block_size = len(trained_blocks)
    trained_arrow_size = len(trained_arrows)

    input_blocks = images[0].contour_list
    input_arrows = arrow_cluster_map
    input_block_size = len(input_blocks)
    input_arrow_size = len(input_arrows)

    block_size_diff = trained_block_size - input_block_size
    arrow_size_diff = trained_arrow_size - input_arrow_size

    if block_size_diff == 0:
        block_thresh, block_list = blockEvaluationNew(trained_blocks, input_blocks, True)
    else:
        block_thresh, block_list = blockEvaluationNew(trained_blocks, input_blocks, False)

    if arrow_size_diff == 0:
        arrow_thresh, arrow_list = arrowEvaluationNew(trained_arrows, input_arrows, True)
    else:
        arrow_thresh, arrow_list = arrowEvaluationNew(trained_arrows, input_arrows, False)

    print block_thresh, block_list
    print arrow_thresh, arrow_list
    X = data.x
    y = data.y

    return predictScoreForDiagram(block_list, arrow_list, block_thresh + arrow_thresh, (X, y))



############# Old logic methods

def diagramEvaluation(image_name_for_evaluation):
    images = [processImage(image_file_dir_path, image_name_for_evaluation)]
    with open(data_save_dir_path + 'trained_data.pkl', 'rb') as f:
        data = dill.load(f)
    cluster_size = len(data.clusters)
    arrow_size = len(data.arrow_cluster_map)
    length = cluster_size + arrow_size
    predict_list = [0] * length

    # Populate initial points for clustering
    init_contour_points = generateInitialClusterPoints(images)

    # Generate some points
    points = generateAllContourPointsForClustering(images)
    # Cluster those data!

    opt_cutoff = 0.5
    clusters = kmeans(init_contour_points, points, opt_cutoff)

    point_cluster_map = kmeansClassification(data.clusters, points)
    evaluateAnswerPoints(point_cluster_map, predict_list)

    # print "Score : ", textScore
    displayClassificationResult(point_cluster_map)

    arrow_cluster_map_old = evaluateArrowClusterRelation(images[0].arrows, data.clusters)
    # (p_new1,clustHead_1_old, clustTail_1), (p_new1,clustHead_1, clustTail_1) ,(p_new1,clustHead_1, clustTail_1), etc
    arrow_cluster_map_new = evaluateArrowClusterRelation(images[0].arrows, clusters)
    # (p_new1,clustHead_1_new, clustTail_1), (arrow,clustHead_1, clustTail_1) ,(p_new1,clustHead_1, clustTail_1), etc
    # print arrow_cluster_map.arr

    prediction_list = []
    prediction_list.append(arrowEvaluation(arrow_cluster_map_old, arrow_cluster_map_new, predict_list, cluster_size))
    predictFromTrainData(prediction_list)
    # arrowScore = evaluateArrowPoints()
    print predict_list


def arrowEvaluation(arrow_cluster_map_old, arrow_cluster_map_new, predict_list, cluster_size):
    score = 0
    lenn = len(arrow_cluster_map_old)

    index = cluster_size
    for (arrow_old, head_cluster_old, tail_cluster_old) in arrow_cluster_map_old:
        for (arrow_new, head_cluster_new, tail_cluster_new) in arrow_cluster_map_new:
            if arrow_old == arrow_new:
                if (((head_cluster_old.points[0].text == head_cluster_new.points[0].text) or (
                semanticWordcheck(head_cluster_new.points[0].text))) and (
                    (tail_cluster_old.points[0].text == tail_cluster_new.points[0].text) or (
                semanticWordcheck(tail_cluster_new.points[0].text)))):
                    predict_list[index] = 100
                    index += 1
                else:
                    predict_list[index] = 0
                    index += 1
    return predict_list


def evaluateAnswerPoints(point_cluster_map, predict_list):
    # Checking whether the answer diagram and trained data has equal no of text blocks/regions or not.
    # check = len(points) - len(point_cluster_map)
    count = 0
    index = 0
    for (point, cluster) in point_cluster_map:
        if cluster.points[0].text == point.text:
            count += 1
            predict_list[index] = 100
        elif point.text != 0:
            flag = semanticWordcheck(point.text)
            if flag == True:
                count += 1
                predict_list[index] = 100
        else:
            count -= 1
            predict_list[index] = 0
        index += 1
    return predict_list


def semanticWordcheck(word):
    dictionaryfile = open('data/rootwords.txt', 'r')
    words = dictionaryfile.read()

    if word in words:
        flag = True
    else:
        flag = False

    return flag
