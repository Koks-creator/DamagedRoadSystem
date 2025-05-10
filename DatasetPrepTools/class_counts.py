from glob import glob
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config


class_file_path = rf"{Config.ROOT_PATH}\train_data\labels\train\classes.txt"
labels_folder = rf"{Config.ROOT_PATH}\train_data\labels\train"

with open(class_file_path) as f:
    classes = f.read().strip().split("\n")

class_count = {class_: 0 for class_ in classes}

all_files = glob(fr"{labels_folder}/*.txt")
for file in all_files:
    if "classes.txt" not in file:
        with open(file) as f:
            cont = f.read().strip().split("\n")
            for con in cont:
                try:
                    class_id = int(con[0])
                except IndexError:
                    print(file)
                    sys.exit()
                class_count[classes[class_id]] += 1

print(class_count)