import os
import mlflow
from ultralytics import YOLO, settings
from itertools import combinations
import time
import yaml
import sys
import torch
import traceback
from mlflow import MlflowClient
import numpy as np
import pickle

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)  # set seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  
        torch.cuda.manual_seed_all(seed_value) 

    # make operations deterministics
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to save results in a YAML file - objective: create personalized graphics
def save_to_yaml(experiment_name, params, metrics, model_info, characteristics, training_time, val_time):       
    data = {
        experiment_name: {
            'params': params,
            'metrics': metrics,
            'model': model_info,
            'characteristics': characteristics,
            'training_time': training_time,
            'val_time': val_time
        }
    }
    # CCreate directory if it doesn't exist
    yaml_dir = os.path.join(os.getcwd(), "yamls_feature_engineering")
    os.makedirs(yaml_dir, exist_ok=True)

    # Create storage path
    yaml_file = os.path.join(yaml_dir, f"{experiment_name}_metrics.yaml")

    # Save content in YAML
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file)

    print(f"Métricas guardadas en: {yaml_file}")

# Function to train, register models in MLFlow and save results in YAML
def train_and_register_models(dataset_index, combinations, model_name, _conf, _iou):
    try:
        # Get dataset config yaml
        dataset_config = os.path.join(os.getcwd(), f"datasets/dataset_{dataset_index}/trainyolo8.yaml")

        # Get specified model
        model = YOLO(f"yolov8{model_name}.pt")  # for example, yolov8n.pt, yolov8s.pt, etc.

        # Set current dir
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Create mlruns folder
        mlruns_path = os.path.join(current_dir, 'mlruns')
        os.makedirs(mlruns_path, exist_ok=True)

        # Configure tracking uri using 'file:///' and mlruns path
        mlflow.set_tracking_uri(f"file://{mlruns_path}")

        # Initialize MlFlowClient
        client = MlflowClient()
        # Name MLFlow experiment
        experiment_name = f"experiment_{dataset_index}_{model_name}_conf_{_conf}_iou_{_iou}"
        # Check if experiment already exists
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            # If it doesn't exists, create it
            experiment_id = client.create_experiment(experiment_name)
        else:
            # If it exists, use it id
            experiment_id = experiment.experiment_id

        params = {
            "dataset": f"dataset_{dataset_index}",
            "model": f"yolov8{model_name}",
            "epochs": 300,
            "imgsz": 640,
            "conf": _conf,
            "iou": _iou,
            "characteristics": combinations
        }

        with mlflow.start_run(experiment_id=experiment_id, nested=True) as run :

            print("Current tracking URI:", mlflow.get_tracking_uri())
            # Train model
            start_time = time.time()
            results = model.train(data=dataset_config, epochs=300, imgsz=640, save=True, plots=True, save_period=50, conf = _conf, iou = _iou)
            end_time = time.time()
            training_time = end_time - start_time
            model_info = {
                "model_name": f"yolov8{model_name}.pt",
                "dataset": dataset_index,
                "mlflow_run_id": run._info._run_id,
                "run_id": str(results.save_dir)
            }
            # Register MLFlow params
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Validate model
            start_time = time.time()
            metrics = model.val(conf = _conf, iou = _iou)
            end_time = time.time()
            val_time = end_time - start_time
            print('before metrics')
            # Register params
            validation_metrics = {
                "train/TP": float(results.confusion_matrix.matrix[0][0]),
                "train/FN": float(results.confusion_matrix.matrix[0][1]),
                "train/FP": float(results.confusion_matrix.matrix[1][0]),
                "train/TN": float(results.confusion_matrix.matrix[1][1]),
                "train/fitness": float(results.fitness),
                "train/speed": results.speed,
                "train/mAP_0.5:0.95": float(results.box.map),
                "train/mAP_0.5": float(results.box.map50),
                "train/mAP_0.75": float(results.box.map75),
                "train/AP50": float(results.box.ap50[0]),
                "train/AP": float(results.box.ap[0]),
                "train/APScores": results.box.all_ap[0].tolist(),
                "train/Precision": float(results.box.p[0]),
                "train/Recall": float(results.box.r[0]),
                "train/F1Score": float(results.box.f1[0]),
                "train/r_curve": results.box.r_curve.tolist(),
                "train/p_curve": results.box.p_curve.tolist(),
                "train/f1_curve": results.box.f1_curve.tolist(),
                "train/prec_values": results.box.prec_values.tolist(),
                "train/curves_RP_recall": results.box.curves_results[0][0].tolist(),
                "train/curves_RP_precision": results.box.curves_results[0][1].tolist(),
                "train/curves_CF_confidence": results.box.curves_results[1][0].tolist(),
                "train/curves_CF_F1": results.box.curves_results[1][1].tolist(),
                "train/curves_CP_confidence": results.box.curves_results[2][0].tolist(),
                "train/curves_CP_precision": results.box.curves_results[2][1].tolist(),
                "train/curves_CR_confidence": results.box.curves_results[3][0].tolist(),
                "train/curves_CR_recall": results.box.curves_results[3][1].tolist(),
                "val/mAP_0.5:0.95": float(metrics.box.map),
                "val/mAP_0.5": float(metrics.box.map50),
                "val/mAP_0.75": float(metrics.box.map75),
                "val/AP50": float(metrics.box.ap50[0]),
                "val/AP": float(metrics.box.ap[0]),
                "val/APScores": metrics.box.all_ap[0].tolist(),
                "val/Precision": float(metrics.box.p[0]),
                "val/Recall": float(metrics.box.r[0]),
                "val/F1Score": float(metrics.box.f1[0]),
                "val/AP_class_index": int(metrics.box.ap_class_index[0]),
                "val/Number_of_classes": metrics.box.nc,
                "val/Fitness": float(metrics.box.fitness())
            }

            print('before yaml')
            # Save in a YAML file
            save_to_yaml(experiment_name, params, validation_metrics, model_info, combinations, training_time, val_time)
            print('before mlflow metrics')
            # Registro de métricas de validación en MLflow
            for metric_name, metric_value in validation_metrics.items():
                # If metric is a dict
                if isinstance(metric_value, dict):
                  for key, value in metric_value.items():
                    if isinstance(value, (np.ndarray, list)):
                       for i, v in enumerate(value):
                             mlflow.log_metric(f"{metric_name}_{key}_{i+1}", float(v))
                    else:
                        mlflow.log_metric(f"{metric_name}_{key}", float(value))
                # If metric is a list
                elif isinstance(metric_value, (np.ndarray, list)):
                  print(f'{metric_value} will not be saved')
                # If metric is a unique value
                else:
                    mlflow.log_metric(metric_name, metric_value)

        print(f"Trainig completed for dataset_{dataset_index} with {model_name} model")

    except Exception as e:
        print("An unexpect error occurred during training or MLFlow registration.")
        print("Error:", str(e))
        print(traceback.format_exc())  # Print traceback

# Get params
if __name__ == "__main__":
    try:
        # Update a setting
        settings.update({"mlflow": False})
        set_seed(42)  # Haz que sea reproducible
        dataset_index = int(sys.argv[1])
        combinations = sys.argv[2]
        model_size = sys.argv[3]
        conf = float(sys.argv[4])
        iou = float(sys.argv[5])

        # Call main function to train and register info
        train_and_register_models(dataset_index, combinations, model_size, conf, iou)

    except Exception as e:
        print("Error in main execution.")
        print("Error:", str(e))
        print(traceback.format_exc())  # Print traceback
