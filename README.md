# Intention_classifier


## Download fine-tuned models

If you want to predict with our models you will download this folder.
Download pre-training models from this google account:

You can use this code in your host environment or in a docker container.

## Requirements
* Python >= 3.10
* Pytorch >= 2.3
* Transformers >= 4.46.3
* Linux OS or Docker

## Installation in docker container
You only need to have docker installed. 
 
### Docker images tested:
* nvcr.io/nvidia/pytorch:24.02-py3

### Create docker container

If you have GPU you will use this command:
```bash
docker run --name name_container -it --net=host --gpus device=device_number -v folder_dir_with_code:/workspace nvcr.io/nvidia/pytorch:24.02-py3 bash
```

If you have not GPU you will use this command:
```bash
docker run --name name_container -it --net=host -v folder_dir_with_code:/workspace nvcr.io/nvidia/pytorch:24.02-py3 bash
```

### Install requirements
```bash
pip install -r requirements.txt
```

## Download dataset
Download the Intent_ES dataset from this link:

## Running the Script

If you want train and predict your model you will use train_predict_model.py
These parameters allow to configure the system to train or predict.


### Required Parameters

| Parameter   | Default Value                        | Description                                                     |
|-------------|--------------------------------------|-----------------------------------------------------------------|
| `config_path` | `/config/intenciones_second.json`    | File path to configuration parameters.                          |
| `model_arg` | `/config/intenciones_second_model.json` | File path to model configuration parameters.                    |
| `model_dir` | `""`                                 | Relative directory of the model for prediction.                 |
| `is_training` | `True`                               | This parameter should be `True` if you want to train the model. |

### Training execution
```bash
python train_predict_model.py --config_path /config/intenciones_second.json --model_arg /config/intenciones_second_model.json --is_training
```

### Prediction execution
```bash
python train_predict_model.py --config_path /config/intenciones_second.json --model_arg /config/intenciones_second_model.json --model_dir /workspace/models/global_intention --is_training False
```