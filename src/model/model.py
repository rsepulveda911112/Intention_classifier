import os
import pandas as pd
import wandb
from simpletransformers.model import ClassificationModel
from common.utils import parse_wandb_param
from sklearn.model_selection import train_test_split
import sklearn


class Model:
    def __init__(self, model_type, model_name, use_cuda, labels_len=None, wandb_project=None, wandb_config=None,
                 is_evaluate=False, best_result_config=None, is_traning=False, output_dir="", model_args=None):
        ############ Hyperparameters #####################
        self.model_args = model_args
        weight = None
        if "weight" in model_args:
            weight = model_args["weight"]
        if is_traning:
            self.is_evaluate = is_evaluate
            self.model_args['evaluate_during_training'] = self.is_evaluate
            self.model_args['wandb_project'] = wandb_project
            self.model_args['output_dir'] = output_dir

            sweep_config = {}
            if wandb_project:
                model_args["weight_decay"] = 0
                wandb.init(config=wandb_config, project=wandb_project)
                wandb_config = wandb.config
                parse_wandb_param(wandb_config, model_args)
            if best_result_config:
                sweep_result = pd.read_csv(os.getcwd() + best_result_config)
                best_params = sweep_result.to_dict()
                parse_wandb_param(best_params, model_args)

            print(model_args)
            self.model = ClassificationModel(model_type, model_name, num_labels=labels_len, use_cuda=use_cuda,
                                             args=model_args, sweep_config=sweep_config, ignore_mismatched_sizes=True, weight=weight)
        else:
            self.model = ClassificationModel(model_type, model_name, use_cuda=use_cuda, num_labels=labels_len, args=model_args,
                                             ignore_mismatched_sizes=True, weight=weight)

    def fit(self, df_train):
        labels = list(df_train['labels'].unique())
        labels.sort()

        df_eval = None
        if self.is_evaluate:
            df_train, df_eval = train_test_split(df_train, test_size=0.2, train_size=0.8, random_state=1)

        self.model.train_model(df_train, eval_df=df_eval, acc=sklearn.metrics.accuracy_score)


    def predict_task(self, df_test):
        if 'text_a' in df_test.columns:
            df_result = pd.concat([df_test['text_a'], df_test['text_b']], axis=1)
        else:
            df_result = pd.concat([df_test['text']], axis=1)
        value_in = df_result.values.tolist()
        y_predict, model_outputs_test = self.model.predict(value_in)
        return y_predict, model_outputs_test
