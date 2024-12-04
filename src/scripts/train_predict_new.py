import argparse
import os
from common.loadData import load_all_data
from common.score import scorePredict
from model.model import Model
import pandas as pd
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight


def main(parser):
    args = parser.parse_args()
    model_dir = args.model_dir
    best_result_config = None
    config_path = args.config_path
    model_arg = args.model_arg
    is_training = args.is_training

    with open(os.getcwd() + config_path) as f:
        config = json.load(f)
    df_model_args = pd.read_json(os.getcwd() + model_arg)
    if "best_result_config" in df_model_args.columns:
        best_result_config = df_model_args['best_result_config'][0]
    model_args = df_model_args.to_dict(orient='records')[0]
    model_name = config["model_name"]
    if model_dir != "":
        model_name = os.getcwd() + model_dir

    df_test, df_test_values, label_encoder = load_all_data(config["test_file"], [], config["label"],
                                            config["filter_label"], config["filter_label_value"],
                                            two_text=config["two_text"], name_text_columns =config["name_text_columns"])
    #
    labels = list(df_test['labels'].unique())
    if is_training:
        df_train, df_train_values, label_encoder = load_all_data(config["train_file"], [], config["label"],
                                                  config["filter_label"], config["filter_label_value"],
                                                  two_text=config["two_text"], name_text_columns =config["name_text_columns"])
        df_train = df_train[0:1000]
        labels = list(df_train['labels'].unique())
        ############### Calculate weights using sklearn ##################
        if "weight" in df_model_args:
            # weights = compute_class_weight(np.unique(df_train['labels'].values), list(df_train['labels'].values))
            weights = compute_class_weight(class_weight="balanced", classes=np.unique(df_train['labels'].values),
                                           y=df_train['labels'].values)
            model_args["weight"] = weights.tolist()

        wandb_config = {}
        stance_model = Model(config["model_type"], model_name, config["use_cuda"], len(labels),
                             config["wandb_project"], wandb_config,
                             config["is_evaluate"], best_result_config, is_training,
                             output_dir=os.getcwd() + '/models/' + config["output_dir"], model_args=model_args)
        stance_model.fit(df_train)

    else:
        stance_model = Model(config["model_type"], model_name, config["use_cuda"], labels_len=len(labels), model_args=model_args)

    y_pred, model_outputs_test = stance_model.predict_task(df_test)
    y_pred = np.argmax(model_outputs_test, axis=1)
    df_pred = pd.DataFrame(y_pred, columns=['labels'])
    df_pred['labels'] = label_encoder.inverse_transform(df_pred['labels'])
    
    df_test['labels'] = label_encoder.inverse_transform(df_test['labels'])
    labels = list(df_test['labels'].unique())
    labels.sort()
    
    result, f1 = scorePredict(df_test['labels'].values, df_pred.values, labels)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--config_path",
                        default="/config/intenciones_second.json",
                        type=str,
                        help="File path to configuration parameters.")
    
    parser.add_argument("--model_arg",
                        default="/config/intenciones_second_model.json",
                        type=str,
                        help="File path to model configuration parameters.")

    parser.add_argument("--model_dir",
                        default="/models/global_intention/",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument("--is_training",
                        default=False,
                        action='store_true',
                        help="This parameter should be True if you use sweep search.")

    main(parser)
