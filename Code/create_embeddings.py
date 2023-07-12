import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import json

from hyperparameters import hyperparams

from transformers import BertTokenizerFast, RobertaTokenizerFast, TFBertModel, TFRobertaModel

def tokenize_fields_contextual(writings_df, tokenizer, model, file):
    embeddings = {}
    iter=0
    for row in tqdm(writings_df.itertuples(), total=writings_df.shape[0]):
        if iter>5000:
            break
        iter+=1
        raw_text = ""
        if hasattr(row, 'title'):
            if row.title and isinstance(row.title, str):
                raw_text += row.title
        if hasattr(row, 'text'):
            if row.text and isinstance(row.text, str):
                raw_text += row.text

        tokenized_post = tokenizer(raw_text,add_special_tokens=True, max_length=hyperparams['maxlen'],
                                        padding='max_length', truncation=True,
                                        return_attention_mask=True,
                                        return_tensors='tf')
        post_ids = tokenized_post['input_ids']
        post_masks = tokenized_post['attention_mask']

        last_four = model(post_ids,
                           attention_mask = post_masks,
                           output_hidden_states=True, return_dict=True)[
                           'hidden_states'][-4:]
        embedding = tf.squeeze(tf.add_n([layer for layer in last_four]), axis=0)

        identifier = row.subject + row.date
        entry = {identifier: embedding.numpy().tolist()}

        with open(file, mode='w', encoding='utf-8') as feedsjson:
            feeds.append(entry)
            json.dump(feeds, feedsjson)


    return embeddings


root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"
task = "Self-Harm"                     #"Self-Harm" - "Anorexia" - "Depression"
embedding_type = "BERT"                #"BERT" - "RoBERTa"

file = "/Users/ronhochstenbach/Desktop/Thesis/Data/Processed Data/embeddings_" + embedding_type + "_" + task + ".json"

writings_df = pd.read_pickle(root_dir + "/Processed Data/df_" + task + ".pkl")

if embedding_type == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)
    model = TFBertModel.from_pretrained('bert-base-uncased')
elif embedding_type == "RoBERTa":
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base',
                                                          do_lower_case=True, add_prefix_space=True)
    model = TFRobertaModel.from_pretrained('roberta-base')
else:
    raise Exception("Unknown model type!")

with open(file, mode='w', encoding='utf-8') as f:
    json.dump([], f)

embeddings_dict = tokenize_fields_contextual(writings_df, tokenizer, model, file)




