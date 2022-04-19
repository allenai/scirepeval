import json
from scidocs.user_activity_and_citations import make_run_from_embeddings, qrel_metrics
from scidocs.embeddings import load_embeddings_from_jsonl
from tqdm import tqdm
from scidocs.classification import classify
from scidocs import get_scidocs_metrics
from scidocs.paths import DataPaths
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.adapt import MLTSVM
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import math
import ast


def read_data(file_path):
    task_data = []
    with open(file_path) as f:
        # [{"corpus_id":...,"title":..., "abstract":...}...]
        try:
            task_data = [json.loads(line) for line in f]
            if len(task_data) == 1:
                task_data = task_data[0]
        except:
            task_data = json.load(f)
    return task_data


def binary_classify(X_train, y_train, X_test, y_test):
    return classify(X_train, y_train, X_test, y_test, avg="binary")


def few_shot_binary(X, y, bm_embeddings, samples, iterations=5):
    f1_scores = []
    skf = StratifiedKFold(n_splits=math.ceil(X.shape[0] / samples))
    count = 0

    for test, train in skf.split(X, y):
        X_train, y_train = np.array([bm_embeddings[d] for d in X[train]]), y[train]
        X_test, y_test = np.array([bm_embeddings[d] for d in X[test]]), y[test]
        f1_scores.append(binary_classify(X_train, y_train, X_test, y_test))
        count += 1
        if count == iterations:
            break
    print(f1_scores)
    return np.mean(f1_scores)


def biomimicry(data_file, embed_file, samples=-1):
    bm_data = read_data(data_file)
    bm_embeddings = load_embeddings_from_jsonl(embed_file)
    X, y = np.array([d["corpus_id"] for d in bm_data if "corpus_id" in d]), np.array(
        [d["label"] for d in bm_data if "corpus_id" in d])
    if samples == -1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
        X_train = np.array([bm_embeddings[d] for d in X_train])
        X_test = np.array([bm_embeddings[d] for d in X_test])
        print("Biomimicry complete dataset F1:{}".format(str(binary_classify(X_train, y_train, X_test, y_test))))
    else:
        print(
            "Biomimicry few shot with {} samples F1:{}".format(samples, few_shot_binary(X, y, bm_embeddings, samples)))


def get_data_from_ids(data_file, multi=False):
    ids, labels = [], []
    with open(data_file, "r") as f:
        d_ids = f.readlines()
    for d in d_ids:
        d = d.strip()
        max_split = -1
        if multi:
            max_split = 1
        d_split = d.split(" ", maxsplit=max_split)
        ids.append(int(d_split[0]))
        if multi:
            labels.append([int(d) for d in ast.literal_eval(d_split[1].replace(" ", ","))])
        else:
            labels.append(int(d_split[1]))
    return np.array(ids), np.array(labels)


def mesh(train_file, test_file, embed_file):
    train_ids, y_train = get_data_from_ids(train_file)
    test_ids, y_test = get_data_from_ids(test_file)
    mesh_embeddings = load_embeddings_from_jsonl(embed_file)
    X_train, X_test = np.array([mesh_embeddings[d] for d in train_ids]), np.array(
        [mesh_embeddings[d] for d in test_ids])
    print("Mesh eval F1: {}".format(classify(X_train, y_train, X_test, y_test, n_jobs=5)))


def fos(train_file, test_file, embed_file):
    train_ids, y_train = get_data_from_ids(train_file, multi=True)
    test_ids, y_test = get_data_from_ids(test_file, multi=True)
    fos_embeddings = load_embeddings_from_jsonl(embed_file)
    X_train, X_test = np.array([fos_embeddings[d] for d in train_ids]), np.array([fos_embeddings[d] for d in test_ids])
    svm = LinearSVC(random_state=42, max_iter=10000)
    Cs = np.logspace(-4, 2, 7)
    svm = GridSearchCV(estimator=svm, cv=3, param_grid={'C': Cs}, n_jobs=5)
    svm = OneVsRestClassifier(svm, n_jobs=4)
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    f1 = np.round(100 * f1_score(y_test, preds, average="macro"), 2)
    print("FoS eval F1: {}".format(f1))


def s2and(qrel_file, run_file, embed_file):
    embeddings = load_embeddings_from_jsonl(embed_file)
    embeddings = {str(k): v for k, v in embeddings.items()}
    make_run_from_embeddings(qrel_file, embeddings, run_file, topk=5, generate_random_embeddings=False)
    coview_results = qrel_metrics(qrel_file, run_file, metrics=('ndcg', 'map'))
    print("S2AND IR eval: " + str(coview_results))


def scidocs(classification_embeddings_path, user_activity_and_citations_embeddings_path, recomm_embeddings_path):
    data_paths = DataPaths("scidocs/data")
    scidocs_metrics = get_scidocs_metrics(
        data_paths,
        classification_embeddings_path,
        user_activity_and_citations_embeddings_path,
        recomm_embeddings_path,
        val_or_test='test',  # set to 'val' if tuning hyperparams
        n_jobs=12,  # the classification tasks can be parallelized
        cuda_device=-1  # the recomm task can use a GPU if this is set to 0, 1, etc
    )

    print(scidocs_metrics)


# for i in range(6):
# biomimicry("biomimicry/test_new.jsonl", "biomimicry/emb_phantasm_class_only_2.json")
# biomimicry("biomimicry/test_new.jsonl", "biomimicry/emb_phantasm_class_only_2.json", samples=64)
# biomimicry("biomimicry/test_new.jsonl", "biomimicry/emb_phantasm_class_only_2.json", samples=16)
s2and("s2and/test.qrel", "s2and/mtl_cls.qrel", f"s2and/emb_phantasm_s2and_search.json")
# s2and("search/test.qrel", "search/mtl_cls.qrel", f"search/emb_phantasm_specter_search.json")
# fos("fos/train_eval_ids.txt", "fos/test_eval_ids.txt", "fos/emb_phantasm_fos_search.json")
# mesh("mesh/train_eval_ids.txt", "mesh/test_eval_ids.txt", "mesh/emb_phantasm_mesh_search.json")
# scidocs("specter2/emb_phantasm_mag_mesh_specter_search.json", "specter2/emb_phantasm_view_cite_read_specter_search.json", "specter2/emb_phantasm_recomm_specter_search.json")















