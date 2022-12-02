"""
conda ML_general

movie_lens 2 approaches.

see tf dataset.
"""
import tensorflow_datasets as tfds
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import os
import requests
import hashlib
import tarfile 
import zipfile
import math
import optuna 

import matplotlib.pyplot as plt


# Download the movielens data

DATA_HUB = dict()

DATA_HUB['ml-100k'] = (
    'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

BASE_DIR = os.getcwd()
np.random.seed(42)

def download(name, cache_dir=os.path.join(BASE_DIR, 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)


def read_data_ml100k():
    data_dir = download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items

# Data inspection

def data_inspect():
    data, num_users, num_items = read_data_ml100k()
    print(num_items, num_users)
    print(data.head())
    print(len(data))

    sparsity = 1 - (len(data)/(num_items*num_items))
    print("sparsity", sparsity)

    # plot rating
    plt.hist(data["rating"], bins=5, ec="black")
    plt.show()
    plt.close()

# Data Preprocessing to get Sequence Aware or random sample of examples

def preprocess(data):
    """transform target for easier trainning. data is a pd"""
    min_rating, max_rating = min(data["rating"].values), max(data["rating"].values)
    print("min_rating", min_rating, "max_rating", max_rating)

    data["rating"] = data["rating"].apply(lambda x: (1.0 * (x - min_rating))/(max_rating - min_rating))
    return data

def split_train_test(data, num_items, num_users, ratio=0.1, mode="random"):
    data = preprocess(data)
    if mode == "seq-aware":
        train_set, test_set, train_list = {}, {}, []

        for line in data.itertuples():
            # print(line)
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_set.setdefault(u, []).append((u, i, rating, time))
            if u not in test_set or test_set[u][-1] < time:
                test_set[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_set[u], key=lambda x: x[3]))
        # print(train_list)
        test_data = [(key, *value) for key, value in test_set.items()]
        train_data = [item for item in train_list if item not in test_data]
        # print(train_data) # this is empty
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        # print(len(test_data))  # here test data is just 944; this is not quite good split code
        train_data.to_csv('movilens/train.csv', index=False)
        test_data.to_csv('movilens/test.csv', index=False)

    else:
        # mode random
        mask = [
            True if x == 1 else False 
            for x in np.random.uniform(0, 1, len(data)) < 1 - ratio
        ]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
        train_data.to_csv('movilens/rand_train.csv', index=False)
        test_data.to_csv('movilens/rand_test.csv', index=False)


    return train_data, test_data



# Get two variants: with explicit or implicit rating

def get_with_feedback(data, num_users, num_items, feedback="explicit"):
    users, items, scores = [], [], []
    inter = np.zeros((num_items+1, num_users+1)) if feedback == "explicit" else {} # no need to do this - implicit can
    # be easily decided later.
    for line in data.itertuples():
        # print(line)
        user_index, item_index = int(line[1]-1), int(line[2]-1) 
        score = line[3] if feedback == "explicit" else 1 # seen
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == "implicit":
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    # print(users[:10])
    return np.hstack((np.array(users).reshape(len(users), -1), np.array(items).reshape(len(items), -1))), np.array(scores), inter


# Implement the embeddings model with Users and Movies embeddings here in a "predict rating"
# can easily add more features to this
# TODO: include a negative sample to do the Implicit feedback variation with binary crossentropy 

class CF(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size, **kwargs):
        super(CF, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding = tf.keras.layers.Embedding(num_users, 
                                                        embedding_size, 
                                                        embeddings_initializer="he_normal",
                                                        embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
                                                        name="user_w_emb"
                                                        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1, name="user_b_emb")

        self.item_embedding = tf.keras.layers.Embedding(num_items,
                                                        embedding_size,
                                                        embeddings_initializer="he_normal",
                                                        embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
                                                        )

        self.item_bias = tf.keras.layers.Embedding(num_items, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0]) # first col in np array
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        tdot = tf.tensordot(user_vector, item_vector, axes=2)
        result = tdot + user_bias + item_bias 
        # squash
        return tf.nn.sigmoid(result)

def run_best_rec():

    # data pipeline
    data, num_users, num_items = read_data_ml100k()
    # split
    train_data, test_data = split_train_test(data, num_items, num_users, ratio=0.1, mode="seq-aware")  

    # input pipeline
    train_features, train_scores, train_inter = get_with_feedback(
        train_data, num_users, num_items, feedback="explicit")
    valid_features, valid_scores, valid_inter = get_with_feedback(
        train_data, num_users, num_items, feedback="explicit")

    # model
    emb_size = 50

    model = CF(num_users, num_items, emb_size)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(), 
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    # train
    history = model.fit(
        x=train_features,
        y=train_scores,
        batch_size=24,
        epochs=1,
        validation_data=(valid_features, valid_scores)
    )

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig("movilens/losses_rec")
    plt.close()

    os.makedirs("rec", exist_ok=True)
    model.save("rec")

    return model



# Tuning

def objective(trial):
    # Clear clutter from previous session graphs.
    tf.keras.backend.clear_session()

    # data pipeline
    data, num_users, num_items = read_data_ml100k()
    # split
    train_data, test_data = split_train_test(data, num_items, num_users, ratio=0.1, mode="seq-aware")  
    # input pipeline
    train_features, train_scores, train_inter = get_with_feedback(
        train_data, num_users, num_items, feedback="explicit")
    valid_features, valid_scores, valid_inter = get_with_feedback(
        train_data, num_users, num_items, feedback="explicit")

    
    # model
    emb_size = trial.suggest_int("EMBEDDING_SIZE", 48, 50, log=True)
    model = CF(num_users, num_items, emb_size)
    

    # We compile our model with a sampled learning rate.
    lr = trial.suggest_float("lr", 1e-5, 1, log=True)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')] # this predicts a score so we have regression
    )


    batch = trial.suggest_int("batch_size", 16, 64)
     # train
    history = model.fit(
        x=train_features,
        y=train_scores,
        batch_size=batch,
        epochs=5,
        validation_data=(valid_features, valid_scores),
        
    )

    res = model.evaluate(valid_features, valid_scores, verbose=0)
    return res[1]


def run_optuna():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(42), direction="minimize")
    study.optimize(objective, n_trials=5, timeout=600)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def retrieve_items_for_user(model):
    testdf= pd.read_csv('movilens/test.csv', names=["user", "item", "rating", "time"])

    # get random user
    user = testdf.user.sample(1).iloc[0]
    print(user)
    # print(testdf.head())

    movies_watched_by_the_user = set(testdf["item"][testdf.user == user].values)
    print(movies_watched_by_the_user) 
    # get all the other movies and then do a model predict
    movies_not_watched = {x for x in testdf["item"].values if x not in movies_watched_by_the_user}
   
    user_col = np.array([user] * len(movies_not_watched)).reshape(-1, 1)
    movies_col = np.array([x for x in movies_not_watched]).reshape(-1, 1)
    predict_input = np.hstack((user_col, movies_col))

    preds = model.predict(predict_input).flatten() * 4 + 1
    print(preds)
    # top 6
    indeces = np.argsort(preds)[::-1][:6]
    print(indeces)
    rec_films = movies_col[indeces]


if __name__ == "__main__":
    
    # hyperpar tune
    # run_optuna()

    # best run
    model = run_best_rec()
    print(model.layers)

    for layer in model.layers:
        print(layer._name) # user_w_emb
        w = layer.get_weights() # list of len1
        print(w[0].shape) # 943, 50; this is the user embedding
        break


    # pred and retrieve engine
    # retrieve_items_for_user(model)
    # Scale predicted scores by the min/max: min = 1; max = 5

