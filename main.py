################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
############## Project Execution begins from here ##############
################################################################

# Imports
from Evaluation import diagramEvaluation
from Training import trainData

if __name__ == "__main__":
    images_names_to_train = ["diag1.jpg"]
    image_name_for_evaluation = "diag1.jpg"
    # trainData(images_names_to_train)
    diagramEvaluation(image_name_for_evaluation)
