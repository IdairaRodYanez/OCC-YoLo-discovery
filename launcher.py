import subprocess
import time
from itertools import combinations

def launch_experiment(index, combinations, model_size, conf, iou):
    try:
        # Call run_yolo_experiments.py with dataset and models params
        subprocess.run(['python3', 'newTrainYolo.py', str(index), combinations, model_size, str(conf), str(iou)], check=True)       
    except subprocess.CalledProcessError as e:
        print(f"Error in dataset_{index} with model {model_size}: {str(e)}")
    print("Waiting 5 minutes before starting next training...")
    time.sleep(900)  # Stop execution 5 minutes (server problems)

if __name__ == "__main__":
    combinations = [('percentil_difference', 'max_peek_dif', 'std'), ('max_peek_dif', 'mean', 'std'), ('max_peek_dif', 'median', 'std')]
    models = ['n', 's', 'm']  # Models that will be trained: yolov8n, yolov8s, yolov8m
    confs = [0.25, 0.1]
    ious = [0.5]
    datasets_indexes = [3, 12, 14]
    i = -1

    for combination in combinations:
      i = i + 1
      for model in models:  
        for iou in ious:
            for conf in confs:
                launch_experiment(datasets_indexes[i, str(combination), model, conf, iou)
