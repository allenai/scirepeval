import pandas as pd
import s3fs
from tqdm import tqdm
import random
from pandarallel import pandarallel
import numpy as np
import itertools

#read raw data from s3
s3 = s3fs.S3FileSystem(anon=False)
#multi-processing for pandas
pandarallel.initialize(progress_bar=True)

def multiple_files_to_df(file_list, description):
    data_list = []
    for file in tqdm(file_list, desc=description):
        tmp_df = pd.read_json(f"s3://{file}", lines=True)
        data_list.append(tmp_df)
    return pd.concat(data_list)

def create_cands_df(pos_df, hneg_df):
    cands_df = pd.concat([pos_df[["pos", "pos_fos", "pyear"]].rename(columns={'pos':'corpus_id', 'pos_fos': 'fos'}), hneg_df[["neg", "neg_fos", "pyear"]].rename(columns={'neg':'corpus_id', 'neg_fos': 'fos'})])
    cands_df = cands_df.drop_duplicates().reset_index(drop=True)

    cands_df["fos"] = cands_df["fos"].apply(lambda x: x.split(","))

    #Group the candidates by FoS into a dict for fast access
    grped_fos_dict = dict()
    for row in tqdm(cands_df.iterrows(), desc="all candidates"):
        row = row[1]
        key = tuple(sorted(row["fos"]))
        if key not in grped_fos_dict:
            grped_fos_dict[key] = []
        grped_fos_dict[key].append(row["corpus_id"])
    grped_fos_dict = {k: np.array(v) for k,v in grped_fos_dict.items()}
    return grped_fos_dict

def get_easy_negs(row):
    query = row["query"]
    pos = row["pos"]
    hnegs = row["hard_negs"]
    q_fos = set(row["query_fos"].split(","))
    rand_negs, same_fos_negs, overlap_fos_negs = [], [], []
    ignore_set = set([query]+pos+hnegs)
    rand_fos_cands = {k:v for k,v in grped_fos_dict.items() if not q_fos.intersection(k)}
    rand_fos_cands = np.setdiff1d(np.unique(np.concatenate([sub_val for sub_val in rand_fos_cands.values()])), ignore_set, assume_unique=True)
    rand_negs = np.random.choice(rand_fos_cands, 2, replace=False).tolist()
    del rand_fos_cands
    
    ignore_set.update(rand_negs)
    overlap_fos_cands = {k:v for k,v in grped_fos_dict.items() if q_fos.intersection(k)}
    overlap_fos_cands = np.setdiff1d(np.unique(np.concatenate([sub_val for sub_val in overlap_fos_cands.values()])), ignore_set, assume_unique=True)
    overlap_fos_negs = np.random.choice(overlap_fos_cands, 2, replace=False).tolist()
    del overlap_fos_cands
    
    ignore_set.update(overlap_fos_negs)
    same_fos_cands = {k:v for k,v in grped_fos_dict.items() if set(k) == q_fos}
    if same_fos_cands:
        same_fos_cands = np.setdiff1d(np.unique(np.concatenate([sub_val for sub_val in same_fos_cands.values()])), ignore_set, assume_unique=True)
        same_fos_negs = np.random.choice(same_fos_cands, 2, replace=False).tolist() if same_fos_cands.size >= 2 else same_fos_cands.tolist()
        
    del same_fos_cands
    
    enegs = rand_negs + same_fos_negs + overlap_fos_negs
    return enegs

def get_triplets(row, metadata):
    triplet_list = []
    query = row["query"]
    pos_set = row["pos"]
    negs_set = set(row["hard_negs"] + row["easy_negs"])
    pos_set = itertools.cycle(pos_set)
    for pos in pos_set:
        if not negs_set:
            break
        neg = negs_set.pop()
        triplet_list.append({"query": metadata[query], "pos": metadata[pos], "neg": metadata[neg]})
    return triplet_list

def get_metadata(queries_df, pos_df, hneg_df):
    metadata = dict()
    for row in tqdm(queries_df.iterrows(), desc="queries metadata"):
        row = row[1]
        metadata[row["corpus_id"]] = {"corpus_id": row["corpus_id"], "title": row["title"].strip(), "abstract": row["abstract"].strip()}
    for row in tqdm(pos_df.iterrows(), desc="positive candidates metadata "):
        row = row[1]
        metadata[row["pos"]] = {"corpus_id": row["pos"], "title": row["title"].strip(), "abstract": row["abstract"].strip()}
    for row in tqdm(hneg_df.iterrows(), desc="negative candidates metadata"):
        row = row[1]
        metadata[row["neg"]] = {"corpus_id": row["neg"], "title": row["title"].strip(), "abstract": row["abstract"].strip()}
    return metadata


if __name__ == '__main__':    
    print("Step 1. Read the +ve and hard -ve candidate json files into individual dataframes and a metadata dict")
    query_files = s3.glob('s3://ai2-s2-aps/specter2_1/queries/*.json')
    pos_files = s3.glob('s3://ai2-s2-aps/specter2_1/pos/*.json')
    neg_files = s3.glob('s3://ai2-s2-aps/specter2_1/negs/*.json')

    queries_df, pos_df, hneg_df = multiple_files_to_df(query_files, "queries"), multiple_files_to_df(pos_files, "positive candidates"), multiple_files_to_df(neg_files, "negative candidates")
    metadata = get_metadata(queries_df, pos_df, hneg_df)

    print("Step 2. Group +ve and hard -ve candidates per query and merge into a single dataframe")
    query_pos_grped = pos_df.groupby(["query", "query_fos"])["pos"].apply(list).reset_index(name='pos')
    query_hnegs_grped = hneg_df.groupby(["query"])["neg"].apply(list).reset_index(name='hard_negs')
    pos_hneg_merged = query_pos_grped.merge(query_hnegs_grped, on=["query"])

    print("Step 3. Merge the +ve and hard -ve candidates for all the queries to create a dataframe of candidates which will be used to sample easy -ves")
    grped_fos_dict = create_cands_df(pos_df, hneg_df)

    print("Step 4. Get a list of upto 6 easy -ves (2 each from same, different and overlapping FoS as the query)")
    pos_hneg_merged["easy_negs"] = pos_hneg_merged.parallel_apply(get_easy_negs, axis=1)

    print("\nStep 5. Create a list of triplets per query from the merged dataframe")
    triplets_list = pos_hneg_merged.parallel_apply(lambda x: get_triplets(x, metadata), axis=1)
    new_triplets = []
    for trips in tqdm(triplets_list, desc="triplets"):
        new_triplets += trips

    triplets_df = pd.DataFrame(new_triplets).sample(frac = 1)
    
    print("Step 6. Writing json dataset back to s3")
    triplets_df.to_json("s3://ai2-s2-aps/specter2_1/new_train.json", orient="records", lines=True)





