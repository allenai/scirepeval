import json
from scidocs.user_activity_and_citations import make_run_from_embeddings, qrel_metrics

from tqdm import tqdm
# from scidocs.classification import classify
from scidocs import get_scidocs_metrics
from scidocs.paths import DataPaths
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.adapt import MLTSVM
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from lightning.classification import LinearSVC
from lightning.regression import LinearSVR
import math
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.dummy import DummyClassifier
from scidocs.embeddings import load_embeddings_from_jsonl

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
from scipy import stats

import json
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def classify(X_train, y_train, X_test, y_test, n_jobs=1, cv=3, avg="macro"):
    """
    Simple classification methods using sklearn framework.
    Selection of C happens inside of X_train, y_train via
    cross-validation.

    Arguments:
        X_train, y_train -- training data
        X_test, y_test -- test data to evaluate on (can also be validation data)

    Returns:
        F1 on X_test, y_test (out of 100), rounded to two decimal places
    """
    estimator = LinearSVC(loss="squared_hinge", random_state=42)
    Cs = np.logspace(-4, 2, 7)
    if cv:
        svm = GridSearchCV(estimator=estimator, cv=cv, param_grid={'C': Cs}, verbose=1, n_jobs=n_jobs)
    else:
        svm = estimator
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    return np.round(100 * f1_score(y_test, preds, average=avg), 2)


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


def binary_classify(X_train, y_train, X_test, y_test, cv=3, n_jobs=1):
    return classify(X_train, y_train, X_test, y_test, n_jobs, cv=cv, avg="binary")


def dummy_classify(X_train, y_train, X_test, y_test, strategy="uniform", avg="binary", cv=None, n_jobs=-1):
    dummy_clf = DummyClassifier(strategy=strategy, random_state=RANDOM_SEED + n_jobs)
    dummy_clf.fit(X_train, y_train)
    preds = dummy_clf.predict(X_test)
    return np.round(100 * f1_score(y_test, preds, average=avg), 2)


def few_shot_binary(X, y, bm_embeddings, samples, iterations=30, random_clf=False):
    f1_scores = []
    # from sklearn.utils import shuffle
    # X, y = shuffle(X, y)
    skf = StratifiedKFold(n_splits=math.ceil(X.shape[0] / samples))
    count = 0
    for test, train in skf.split(X, y):
        X_train, y_train = np.array([bm_embeddings[d] for d in X[train]]), y[train]
        X_test, y_test = np.array([bm_embeddings[d] for d in X[test]]), y[test]
        classify_fn = binary_classify if not random_clf else dummy_classify
        f1 = classify_fn(X_train, y_train, X_test, y_test, cv=None, n_jobs=1 if not random_clf else count)
        f1_scores.append(f1)
        count += 1
        if count == iterations:
            break
    print(f1_scores)
    return np.mean(f1_scores)


def biomimicry(data_file, embed_file, samples=-1, random_clf=False):
    bm_data = read_data(data_file)
    bm_embeddings = load_embeddings_from_jsonl(embed_file)
    X, y = np.array([d["corpus_id"] for d in bm_data if "corpus_id" in d]), np.array(
        [d["label"] for d in bm_data if "corpus_id" in d])
    if samples == -1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
        X_train = np.array([bm_embeddings[d] for d in X_train])
        X_test = np.array([bm_embeddings[d] for d in X_test])
        classify_fn = binary_classify if not random_clf else dummy_classify
        print("Biomimicry complete dataset F1:{}".format(str(classify_fn(X_train, y_train, X_test, y_test))))
    else:
        print("Biomimicry few shot with {} samples F1:{}".format(samples, few_shot_binary(X, y, bm_embeddings, samples,
                                                                                          50 if samples == 64 else 100,
                                                                                          random_clf)))


def get_data_from_ids(data_file, multi=False, typ=int):
    ids, labels = [], []
    with open(data_file, "r") as f:
        d_ids = f.readlines()
    for d in d_ids:
        d = d.strip()
        max_split = -1
        if multi:
            max_split = 1
        d_split = d.split(maxsplit=max_split)
        try:
            ids.append(int(d_split[0]))
        except:
            ids.append(d_split[0])
        if multi:
            labels.append([int(d) for d in ast.literal_eval(d_split[1].replace(" ", ","))])
        else:
            labels.append(typ(d_split[1]))
    return np.array(ids), np.array(labels)


def few_shot_multi(X, y, X_test, y_test, clf_embeddings, samples, iterations=30, random_clf=False):
    f1_scores = []
    skf = StratifiedKFold(n_splits=math.ceil(X.shape[0] / samples))
    count = 0
    X_test = np.array([clf_embeddings[d] for d in X_test])
    for _, train in skf.split(X, y):
        X_train, y_train = np.array([clf_embeddings[d] for d in X[train]]), y[train]
        classify_fn = classify if not random_clf else dummy_classify
        f1_scores.append(classify_fn(X_train, y_train, X_test, y_test, cv=None, avg="macro", n_jobs=5))
        count += 1
        if count == iterations:
            break
    print(f1_scores)
    return np.mean(f1_scores)


def multi_class_clf(train_file, test_file, embed_file, task_name, samples=-1, random_clf=False):
    train_ids, y_train = get_data_from_ids(train_file)
    test_ids, y_test = get_data_from_ids(test_file)
    embeddings = load_embeddings_from_jsonl(embed_file)
    if samples == -1:
        X_train, X_test = np.array([embeddings[d] for d in train_ids]), np.array([embeddings[d] for d in test_ids])
        classify_fn = classify if not random_clf else dummy_classify
        print("{} eval F1: {}".format(task_name, classify_fn(X_train, y_train, X_test, y_test, avg="macro", n_jobs=5)))
    else:
        print("{} few shot with {} samples F1:{}".format(task_name, samples,
                                                         few_shot_multi(train_ids, y_train, test_ids, y_test,
                                                                        embeddings, samples,
                                                                        50 if samples == 64 else 100,
                                                                        random_clf=random_clf)))


def dummy_multi_label(X_train, y_train, X_test, y_test, strategy="uniform"):
    preds_list = []
    for i in range(y_train.shape[1]):
        dummy_clf = DummyClassifier(strategy=strategy, random_state=RANDOM_SEED + i)
        dummy_clf.fit(X_train, y_train[:, i])
        preds = dummy_clf.predict(X_test)
        preds_list.append(preds)
    preds = np.transpose(np.vstack(preds_list))
    return np.round(100 * f1_score(y_test, preds, average="macro"), 2)


def multi_label_clf(X_train, y_train, X_test, y_test):
    svm = LinearSVC(max_iter=10000)
    Cs = np.logspace(-4, 2, 7)
    svm = GridSearchCV(estimator=svm, cv=3, param_grid={'C': Cs}, n_jobs=5)
    svm = OneVsRestClassifier(svm, n_jobs=4)
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    with open("phantasm_new/sample_data/fos_labels.txt", "r") as f:
        labels = f.readlines()
    labels = [l.strip() for l in labels]
    # print(classification_report(y_test, preds, target_names=labels))

    f1 = np.round(100 * f1_score(y_test, preds, average="macro"), 2)
    return f1


def few_shot_multi_label(X, y, X_test, y_test, clf_embeddings, samples, iterations=30):
    f1_scores = []
    X_test = np.array([clf_embeddings[d] for d in X_test])
    for k in tqdm(range(iterations)):
        idx_set = set()
        np.random.seed(RANDOM_SEED + k)
        for yi in range(y.shape[1]):
            idx_set.update(np.random.choice(np.where(y[:, yi] == 1)[0], samples, replace=False).tolist())
        req_idx = list(idx_set)
        X_train, y_train = np.array([clf_embeddings[d] for d in X[req_idx]]), y[req_idx]
        f1_scores.append(multi_label_clf(X_train, y_train, X_test, y_test))
        if not k:
            print(y_train.shape)
    print(f1_scores)
    np.random.seed(RANDOM_SEED)
    return np.mean(f1_scores)


def fos(train_file, test_file, embed_file, samples=-1, random_clf=False):
    train_ids, y_train = get_data_from_ids(train_file, multi=True)
    test_ids, y_test = get_data_from_ids(test_file, multi=True)
    fos_embeddings = load_embeddings_from_jsonl(embed_file)
    if samples == -1:
        X_train, X_test = np.array([fos_embeddings[d] for d in train_ids]), np.array(
            [fos_embeddings[d] for d in test_ids])
        classify_fn = multi_label_clf if not random_clf else dummy_multi_label
        print("FoS eval F1: {}".format(classify_fn(X_train, y_train, X_test, y_test)))
    else:
        print("FoS few shot with {} samples/class F1:{}".format(samples,
                                                                few_shot_multi_label(train_ids, y_train, test_ids,
                                                                                     y_test, fos_embeddings, samples,
                                                                                     50 if samples == 64 else 100)))


def ir(qrel_file, run_file, embed_file, task, k=5, metrics=('ndcg', 'map')):
    embeddings = load_embeddings_from_jsonl(embed_file)
    embeddings = {str(k): v for k, v in embeddings.items()}
    make_run_from_embeddings(qrel_file, embeddings, run_file, topk=k, generate_random_embeddings=False)
    coview_results = qrel_metrics(qrel_file, run_file, metrics=metrics)
    print(f"{task} IR eval: " + str(coview_results))


def peer_review(qc_pairs, peer_review_embeddings, reviewer_metadata):
    sub_embeddings = np.array([peer_review_embeddings[q] for q in qc_pairs])
    reviewers = list(filter(lambda x: x["papers"], reviewer_metadata))
    reviewers_idx = {r["r_id"]: i for i, r in enumerate(reviewers)}
    avg_scores = []
    for r in tqdm(reviewers):
        r_id = r["r_id"]
        p_ids = r["papers"]
        embedding_stack = np.array([peer_review_embeddings[p] for p in p_ids])
        sim_score = cosine_similarity(sub_embeddings, embedding_stack)
        avg_sim_score = np.mean(-np.sort(-sim_score, axis=1)[:, :3], axis=1)
        avg_scores.append(avg_sim_score)
    agg_sim_scores = np.column_stack(avg_scores)
    top10_raw = []
    for i, q in enumerate(qc_pairs):
        c = qc_pairs[q]
        req_idx = []
        for cand in c:
            if cand in reviewers_idx:
                req_idx.append(reviewers_idx[cand])
        scores = agg_sim_scores[i][req_idx]
        scores_dict = {req_idx[i]: scores[i] for i in range(len(req_idx))}
        sorted_scores = dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True))
        for j, idx in enumerate(sorted_scores):
            if j < 10:
                pred_10 = "1"
            else:
                pred_10 = "0"
            top10_raw.append(f"{q} 0 {reviewers[idx]['r_id']} {pred_10} {sorted_scores[idx]} n/a\n")
    return top10_raw


def get_peer_review_params(qrel_file, embed_file, reviewer_metadata):
    f1 = open(qrel_file.format("soft"), "r")
    lines = [l.split() for l in f1.readlines()]
    qc_pairs = dict()
    for l in lines:
        q = l[0]
        if q not in qc_pairs:
            qc_pairs[q] = []
        qc_pairs[q].append(int(l[2]))
    review_embeddings = {str(k): v for k, v in load_embeddings_from_jsonl(embed_file).items()}
    reviewer_metadata = read_data(reviewer_metadata)
    return qc_pairs, review_embeddings, reviewer_metadata


def peer_review_results(qrel_file, run_file, task, metrics=("map", "P_5", "P_10")):
    print(task)
    print(qrel_metrics(qrel_file.format("hard"), run_file, metrics))
    print(qrel_metrics(qrel_file.format("soft"), run_file, metrics))


def combined_peer_review(nips_qrel_file, combined_qrel_file, nips_embed_file, icip_embed_file, nips_reviewer_metadata,
                         icip_reviewer_metadata):
    nips_qc_pairs, nips_peer_review_embeddings, nips_reviewer_metadata = get_peer_review_params(nips_qrel_file,
                                                                                                nips_embed_file,
                                                                                                nips_reviewer_metadata)
    nips_result = peer_review(nips_qc_pairs, nips_peer_review_embeddings, nips_reviewer_metadata)
    with open("peer_review/top_10_nips_pred.qrel", "w") as f:
        f.writelines(nips_result)
    peer_review_results(nips_qrel_file, "peer_review/top_10_nips_pred.qrel", "nips peer review")

    icip_qc_pairs, icip_peer_review_embeddings, icip_reviewer_metadata = get_peer_review_params(combined_qrel_file,
                                                                                                icip_embed_file,
                                                                                                icip_reviewer_metadata)
    icip_qc_pairs = {k: v for k, v in icip_qc_pairs.items() if k not in nips_qc_pairs}
    icip_results = peer_review(icip_qc_pairs, icip_peer_review_embeddings, icip_reviewer_metadata)
    combined_result = nips_result + icip_results
    with open("peer_review/top_10_pred.qrel", "w") as f:
        f.writelines(combined_result)
    print("combined peer review")
    peer_review_results(combined_qrel_file, "peer_review/top_10_pred.qrel", "combined peer review")


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


def s2and_mini(embedding_suffix, block_path):
    import os
    cmd = f"cd S2AND; python3 scripts/custom_block_transfer_experiment_seed_paper.py --custom_block_path {block_path} --experiment_name mini_customblock_phantasm_v1 --exclude_medline --emb_suffix _{embedding_suffix}.pkl"
    os.system(cmd)


def regression(train_file, test_file, embeddings, random_clf):
    train_ids, y_train = get_data_from_ids(train_file, typ=float)
    test_ids, y_test = get_data_from_ids(test_file, typ=float)
    if random_clf:
        preds = np.random.uniform(np.min(y_train), np.max(y_train), y_test.shape)
    else:
        if type(embeddings) == str:
            embeddings = load_embeddings_from_jsonl(embeddings)
        X_train, X_test = np.array([embeddings[d] for d in train_ids]), np.array([embeddings[d] for d in test_ids])
        svm = LinearSVR(random_state=42)
        Cs = np.logspace(-4, 2, 7)
        svm = GridSearchCV(estimator=svm, cv=3, param_grid={'C': Cs}, verbose=1, n_jobs=5)
        svm.fit(X_train, y_train)
        preds = svm.predict(X_test)
    return y_test, preds


def pred_reg(train_file, test_file, task_name, embed_file=None, embeddings=None, random_clf=False):
    y_test, preds = regression(train_file, test_file, embeddings if embeddings else embed_file, random_clf)
    mse = mean_squared_error(y_test, preds)
    r_square = r2_score(y_test, preds)
    tau = stats.kendalltau(y_test, preds)[0]
    # pearson = np.corrcoef(y_test, preds)[0,1]
    print("{} eval MSE: {}, R2: {}, Kendell's Tau: {}".format(task_name, mse, r_square, tau))
    # print(f"Pearson's: {pearson}")


def pred_rating_hIndex(data_dir, tasks, embed_file):
    embeddings = load_embeddings_from_jsonl(embed_file)
    for t, pre in tasks.items():
        pred_reg(f"{data_dir}/{pre}_train.txt", f"{data_dir}/{pre}_test.txt", t, embeddings=embeddings)


# for i in range(6):
suffix_set = ["scincl_ctrl"]
for suffix in suffix_set:
    print("=============================================================")
    print(suffix)
    print("=============================================================")
    biomimicry("biomimicry/test_new.jsonl", f"biomimicry/emb_phantasm_{suffix}.json", random_clf=False)
    biomimicry("biomimicry/test_new.jsonl", f"biomimicry/emb_phantasm_{suffix}.json", samples=64, random_clf=False)
    biomimicry("biomimicry/test_new.jsonl", f"biomimicry/emb_phantasm_{suffix}.json", samples=16, random_clf=False)
    multi_class_clf("drsm/train_eval_id2.txt", "drsm/test_eval_id2.txt", f"drsm/emb_phantasm_{suffix}.json", "DRSM",
                    random_clf=False)
    multi_class_clf("drsm/train_eval_id2.txt", "drsm/test_eval_id2.txt", f"drsm/emb_phantasm_{suffix}.json", "DRSM",
                    samples=64, random_clf=False)
    multi_class_clf("drsm/train_eval_id2.txt", "drsm/test_eval_id2.txt", f"drsm/emb_phantasm_{suffix}.json", "DRSM",
                    samples=24, random_clf=False)
    ir("feeds/paper_query.qrel", "feeds/mtl_paper_query.qrel", f"feeds/emb_phantasm_paper_query_{suffix}.json",
       "feed paper query")
    ir("feeds/multi_paper_query.qrel", "feeds/mtl_multi_aper_query.qrel",
       f"feeds/emb_phantasm_multi_paper_query_{suffix}.json", "feed multi paper query")
    ir("feeds/name_query.qrel", "feeds/mtl_name_query.qrel", f"feeds/emb_phantasm_name_query_{suffix}.json",
       "feed title query")
    ir("trec_covid/test.qrel", "trec_covid/mtl.qrel", f"trec_covid/emb_phantasm_{suffix}.json", "trec_covid")
    combined_peer_review("peer_review/nips_2006_{}.qrel", "peer_review/test_complete_{}.qrel",
                         f"peer_review/emb_phantasm_nips_{suffix}.json", f"peer_review/emb_phantasm_icip_{suffix}.json",
                         "peer_review/nips_reviewer_papers_complete.json",
                         "peer_review/icip_reviewer_papers_complete.json")
    pred_rating_hIndex("rating_hIndex", {"ICLR Ratings": "iclr", "NeuRIPS Ratings": "neurips", "h-Index": "hidx"},
                       f"rating_hIndex/emb_phantasm_{suffix}.json")
    pred_reg("tweet_mentions/train.txt", "tweet_mentions/test.txt", "Tweet Mentions Count",
             f"tweet_mentions/emb_phantasm_{suffix}.json")
    scidocs(f"specter2/emb_phantasm_mag_mesh_{suffix}.json", f"specter2/emb_phantasm_view_cite_read_{suffix}.json",
            f"specter2/emb_phantasm_recomm_{suffix}.json")
    ir("s2and/test_new.qrel", "s2and/mtl_new.qrel", f"s2and/emb_phantasm_{suffix}.json", "s2and")
    ir("search/test.qrel", "search/mtl_cls.qrel", f"search/emb_phantasm_{suffix}.json", "search")
    ir("cite_context/test.qrel", "cite_context/mtl_cls.qrel", f"cite_context/emb_phantasm_{suffix}.json",
       "cite context")
    fos("fos/train_eval_ids.txt", "fos/gold_test_eval_ids.txt", f"fos/emb_phantasm_{suffix}.json", random_clf=False)
    fos("fos/train_eval_ids.txt", "fos/gold_test_eval_ids.txt", f"fos/emb_phantasm_{suffix}.json", samples=5)
    fos("fos/train_eval_ids.txt", "fos/gold_test_eval_ids.txt", f"fos/emb_phantasm_{suffix}.json", samples=10)
    pred_reg("reg_cite_count/train_val_ids.txt", "reg_cite_count/test_val_ids2.txt", "Citation Count Prediction",
             f"reg_cite_count/emb_phantasm_{suffix}.json", )
    pred_reg("reg_yr_ath/train_val_ids2.txt", "reg_yr_ath/test_val_ids2.txt", "Year Prediction",
             f"reg_yr_ath/emb_phantasm_{suffix}.json")
    multi_class_clf("mesh/train_eval_ids.txt", "mesh/test_eval_ids.txt", f"mesh/emb_phantasm_{suffix}.json", "MeSH",
                    random_clf=False)
    s2and_mini(suffix, "/net/nfs2.s2-research/scidocs/data/s2and/full_test_partitioned")


















