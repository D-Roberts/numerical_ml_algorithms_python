import datasets
from datasets import load_dataset
import json

import tensorflow as tf 
import transformers 
from transformers import AutoTokenizer, TFBertForSequenceClassification, set_seed
from transformers.optimization_tf import AdamWeightDecay

import string

import pandas as pd 
import numpy as np 
import os 
import random 

random.seed(42)
np.random.seed(42)
set_seed(42)


def remove_pct(x):
    puncts = list(string.punctuation)
    for y in puncts:
        x = x.replace(y, "")
    x = x.replace("\n", "").replace("\t", "")
    return x

class DataProcessor:
    def __init__(self):
        pass

    def load(self):
        # dataset = load_dataset("reddit", cache_dir=".", data_dir=".") # conect eror
        pass

    def preprocess(self):
        data = pd.read_csv("red.csv", engine="python")
        # print(data.head())
        # print(data.isna().sum())
        # uniqueLabels, c = np.unique(data["label"], return_counts=True)
        # subsample 1200 per category
        data_sample = data.groupby("label", 
            group_keys=True).apply(lambda x: x.sample(min(1200, len(x)), random_state=42))

        print(data_sample.label.value_counts())

        data_sample = data_sample.dropna()
        data_sample["sentence1"] = data_sample["sentence1"].apply(lambda x: remove_pct(x))
        # print(data_sample.head())
        label_list = list(set(data["label"].values))
        label_list.sort()
        label2ind = {k:v for v, k in enumerate(label_list)}
        with open("label_map.json", "w") as fp:
            json.dump(label2ind, fp)

        data_sample["label"] = data_sample["label"].apply(lambda x: label2ind[x])

        data_sample.to_csv("preprocessed.csv", header = ["sentence1", "label"], columns=["sentence1", "label"],
            index=False)

        
        return data_sample, label2ind

    def get_train_valid(self):
        data = pd.read_csv("preprocessed.csv", usecols=["sentence1", "label"], engine="python")
        data = data.sample(frac=1, random_state=42)
        nsize = len(data)
        ntrain = int(0.8 * nsize)
        features, labels = data["sentence1"].values.astype("str"),  data["label"].values.astype(np.int32)
        return  features[:ntrain], labels[:ntrain], features[ntrain:], labels[ntrain:]
           

class Batches(tf.keras.utils.Sequence):
    def __init__(self, trainf, trainl, add_labels=True, shuffle=True, batch_size=32):
        self.features = trainf
        self.labels = trainl 
        self.add_labels = add_labels
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.indexes = np.arange(len(self.features))

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        self.on_epoch_end()

    def __len__(self):
        return len(self.features) // self.batch_size

    def __getitem__(self, idx):
        batch_inds = self.indexes[idx * self.batch_size: (idx+1) * self.batch_size]
        batch_features = self.features[batch_inds]

        if self.add_labels:
            batch_labels = self.labels[batch_inds]

        encoded = self.tokenizer.batch_encode_plus(batch_features.tolist(),
            padding=True, 
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="tf",
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True)

        
        input_ids = np.array(encoded["input_ids"], dtype=np.int32)
        attention_mask = np.array(encoded["attention_mask"], dtype=np.int32)
        token_type_ids = np.array(encoded["token_type_ids"], dtype=np.int32)

        feat_indeces = [input_ids, attention_mask, token_type_ids]
        
        if self.add_labels:
            return feat_indeces, np.array(batch_labels, dtype=np.int32)

    def on_epoch_end(self):
        if self.shuffle:
            random.Random(42).shuffle(self.indexes)


BATCH = 32
LR = 2e-5
EPOCH = 1
MAX_LENGTH = 128


def run_train_eval():
    trainf, trainl, validf, validl = dp.get_train_valid()
    with open("label_map.json", "r") as fp:
        label2ind = json.load(fp)

    btrain = Batches(trainf, trainl, add_labels=True, shuffle=True, batch_size=BATCH)
    bvalid = Batches(validf, validl, add_labels=True, shuffle=True, batch_size=BATCH)

    ind2label = {v:k for k, v in label2ind.items()}

    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(ind2label.keys()))
    model.config.label2ind = label2ind
    model.config.ind2label = ind2label

    num_steps = EPOCH * len(btrain)

    opt = AdamWeightDecay(tf.keras.optimizers.schedules.PolynomialDecay(LR, decay_steps = num_steps), 
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    model.compile(opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    hist = model.fit(btrain, validation_data=bvalid, epochs=EPOCH)

    # print(model.evaluate(bvalid))

# if __name__ == "__main__":
# run_train()
# print(dir(datasets))

# print(help(datasets.load_dataset))
# dp = DataProcessor()
# dp.load()
# dp.preprocess()

# run_train_eval()

