# Delete results
import os

def deleteTrainingResults(map: str, mdl: str, observation: str):
    current_directory = os.getcwd()
    new_directory = current_directory + "/results/train/"
    file_beginning = f"{map}-{mdl}-{observation}"
    files = os.listdir(new_directory)
    for file in files:
        if file.startswith(file_beginning):
            file_path = os.path.join(new_directory, file)
            os.remove(file_path)
            print(f"Deleted {file}")

def deleteSimulationResults(map: str, mdl: str, observation: str):
    current_directory = os.getcwd()
    new_directory = current_directory + "/results/sim/"
    file_beginning = f"{map}-{mdl}-{observation}"
    files = os.listdir(new_directory)
    for file in files:
        if file.startswith(file_beginning):
            file_path = os.path.join(new_directory, file)
            os.remove(file_path)
            print(f"Deleted {file}")
