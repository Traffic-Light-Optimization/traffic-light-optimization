# Delete results
import os
import re

def deleteResults():
  current_directory = os.getcwd()
  new_directory = current_directory + "/results/train/"
  files = os.listdir(new_directory)
  pattern = r'^results.*\.csv$'
  # Delete files matching the pattern
  for file in files:
      if re.match(pattern, file):
          file_path = os.path.join(new_directory, file)
          os.remove(file_path)
      print("Deleted results")