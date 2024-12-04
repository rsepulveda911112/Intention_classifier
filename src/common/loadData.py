import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re


def de_emojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F92F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001F190-\U0001F1FF"
                                        u"\U0001F926-\U0001FA9F"
                                        u"\u2640-\u2642"
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\ufe0f"
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(value):
    new_value = de_emojify(value)
    new_value = re.sub(r'http\S+', '', new_value)
    return new_value


def load_all_data(file_in, labels_to_exclude, label, filter_label, filter_label_value, is_preprocess=False, two_text=False, name_text_columns=None):
    if file_in.endswith('.tsv'):
        df_in = pd.read_csv(os.getcwd() + file_in, sep='\t')
    elif file_in.endswith('.csv'):
        df_in = pd.read_csv(os.getcwd() + file_in)
    else:
        df_in = pd.read_json(os.getcwd() + file_in, orient='records')

    for value in labels_to_exclude:
        df_in = df_in[df_in[label] != value]
    if label in df_in.columns:
        if filter_label:
            df_in = df_in[df_in[filter_label] == filter_label_value]
        print(df_in[label].value_counts())
        labels = df_in[label]
        # To labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels.values)
        print(label_encoder.classes_)
        class_dict = {label_encoder.classes_[i]: i for i in range(len(label_encoder.classes_))}
        print(class_dict)

    if not two_text:
        if is_preprocess:
            df_in[name_text_columns[0]] = df_in[name_text_columns[0]].apply(preprocess)
        list_of_tuples = list(zip(list(df_in[name_text_columns[0]]), list(labels)))
        df = pd.DataFrame(list_of_tuples, columns=['text', 'labels'])
    else:
        if is_preprocess:
            for name_column in name_text_columns:
                df_in[name_column] = df_in[name_column].apply(preprocess)
        list_of_tuples = list(zip(list(df_in[name_text_columns[0]]), list(df_in[name_text_columns[1]]), list(labels)))
        df = pd.DataFrame(list_of_tuples, columns=['text_a', 'text_b', 'labels'])
    return df, df_in, label_encoder
