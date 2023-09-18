# Delete results
import os

def deleteTrainingResults(map: str, type: str, mdl: str):
    current_directory = os.getcwd()
    new_directory = current_directory + "/results/train/"
    file_beginning = f"results-{map}-{type}-{mdl}"
    files = os.listdir(new_directory)
    for file in files:
        if file.startswith(file_beginning):
            file_path = os.path.join(new_directory, file)
            os.remove(file_path)
            print(f"Deleted {file}")

def deleteSimulationResults(map: str, type: str, mdl: str):
    current_directory = os.getcwd()
    new_directory = current_directory + "/results/sim/"
    file_beginning = f"results_sim-{map}-{type}-{mdl}"
    files = os.listdir(new_directory)
    for file in files:
        if file.startswith(file_beginning):
            file_path = os.path.join(new_directory, file)
            os.remove(file_path)
            print(f"Deleted {file}")

def deleteRandResults(map: str, type: str, mdl: str):
    current_directory = os.getcwd()
    new_directory = current_directory + "/results/rand/"
    file_beginning = f"results_rand-{map}-{type}-{mdl}"
    files = os.listdir(new_directory)
    for file in files:
        if file.startswith(file_beginning):
            file_path = os.path.join(new_directory, file)
            os.remove(file_path)
            print(f"Deleted {file}")