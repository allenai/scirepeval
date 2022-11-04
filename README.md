# SciRepEval: A Multi-Format Benchmark for Scientific Document Representations
This repo contains the code to train, evaluate and reproduce the representation learning models and results on the benchmark introduced in [SciRepEval](https://openreview.net/pdf?id=zfiYcbeQkH).

## Quick Setup
Clone the repo and setup the environment as follows:
```bash
git clone git@github.com:allenai/scirepeval.git
cd scirepeval
conda create -n scirepeval python=3.8
conda activate scirepeval
pip install -r requirements.txt
```
## Usage
Please refer to the following for further usage:

[Training](https://github.com/allenai/scirepeval/blob/main/training/TRAINING.MD)

[Benchmarking](https://github.com/allenai/scirepeval/blob/main/BENCHMARKING.md)

## Benchmark Details
SciRepEval consists of 25 scientific document tasks to train and evaluate scientific document representation models. The tasks are divided across 4 task formats- classification **CLF**, regression **RGN**, proximity (nearest neighbors) retrieval **PRX** and ad-hoc search **SRCH**.  The table below gives a brief overview of the tasks with their HuggingFace datasets config names, if applicable. 
The benchmark dataset can be downloaded from AWS S3 or HuggingFace as follows:
#### AWS S3 via CLI
```bash
mkdir scirepeval_data && scirepeval_data
aws s3 sync s3://ai2-s2-research-public/scirepeval/train .
aws s3 sync s3://ai2-s2-research-public/scirepeval/test .
```
The AWS CLI commands can be run with the `--dryrun`  flag to list the files being copied. The entire dataset is ~24 GB in size.

#### HuggingFace Datasets
The training, validation and raw evaluation data is available at [allenai/scirepeval](https://huggingface.co/datasets/allenai/scirepeval), while the labelled test examples are available at [allenai/scirepeval_test](https://huggingface.co/datasets/allenai/scirepeval_test).

```python
import datasets
dataset = datsets.load_dataset(<dataset name>, <hf config name>)
```

Since we want to evaluate document representations, every dataset consists of two parts: test metadata (text for representation generation available under allenai/scirepeval) and labelled examples (available under allenai/scirepeval_test)

|Format|Name|Train|Metric|HF Config| HF Test Config|
|--|--|--|--|--|--|
|CLF|[MeSH Descriptors](https://www.nlm.nih.gov/databases/download/terms_and_conditions_mesh.html)|Y|F1 Macro|[mesh_descriptors](https://huggingface.co/datasets/allenai/scirepeval/viewer/mesh_descriptors)|[mesh_descriptors](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/mesh_descriptors)|
|CLF|Fields of study|Y|F1 Macro|[fos](https://huggingface.co/datasets/allenai/scirepeval/viewer/fos)|[fos](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/fos)|
|CLF|[Biomimicry](https://github.com/nasa-petal/PeTaL-db)|N|F1 Binary|[biomimicry](https://huggingface.co/datasets/allenai/scirepeval/viewer/biomimicry)|[biomimicry](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/biomimicry)|
|CLF|[DRSM](https://github.com/chanzuckerberg/DRSM-corpus)|N|F1 Macro|[drsm](https://huggingface.co/datasets/allenai/scirepeval/viewer/drsm)|[drsm](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/drsm)|
|CLF|[SciDocs-MAG](https://github.com/allenai/scidocs)|N|F1 Macro|[scidocs_mag_mesh](https://huggingface.co/datasets/allenai/scirepeval/viewer/scidocs_mag_mesh)|[scidocs_mag](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/scidocs_mag)|
|CLF|[SciDocs-Mesh Diseases](https://github.com/allenai/scidocs)|N|F1 Macro|[scidocs_mag_mesh](https://huggingface.co/datasets/allenai/scirepeval/viewer/scidocs_mesh)|[scidocs_mesh](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/scidocs_mag_mesh)|
|RGN|Citation Count|Y|Kendall's Tau|[cite_count](https://huggingface.co/datasets/allenai/scirepeval/viewer/cite_count)|[cite_count](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/cite_count)|
|RGN|Year of Publication|Y|Kendall's Tau|[pub_year](https://huggingface.co/datasets/allenai/scirepeval/viewer/pub_year)|[pub_year](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/pub_year)|
|RGN|[Peer Review Score](https://api.openreview.net)|N|Kendall's Tau|[peer_review_score_hIndex](https://huggingface.co/datasets/allenai/scirepeval/viewer/peer_review_score_hIndex)|[peer_review_score](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/peer_review_score)|
|RGN|[Max Author hIndex](https://api.openreview.net)|N|Kendall's Tau|[peer_review_score_hIndex](https://huggingface.co/datasets/allenai/scirepeval/viewer/peer_review_score_hIndex)|[hIndex](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/hIndex)|
|RGN|[Tweet Mentions](https://github.com/lingo-iitgn/TweetPap)|N|Kendall's Tau|[tweet_mentions](https://huggingface.co/datasets/allenai/scirepeval/viewer/tweet_mentions)|[tweet_mentions](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/tweet_mentions)|
|PRX|Same Author Detection|Y|MAP|[same_author](https://huggingface.co/datasets/allenai/scirepeval/viewer/same_author)|[same_author](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/same_author)|
|PRX|Highly Influential Citations|Y|MAP|[high_influence_cite](https://huggingface.co/datasets/allenai/scirepeval/viewer/high_influence_cite)|[high_influence_cite](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/high_influence_cite)|
|PRX|Citation Prediction|Y|-|[cite_prediction](https://huggingface.co/datasets/allenai/scirepeval/viewer/cite_prediction)|-|
|PRX|S2AND*|N|B^3 F1|-|-|
|PRX|Paper-Reviewer Matching**|N|Precision@5,10|[paper_reviewer_matching](https://huggingface.co/datasets/allenai/scirepeval/viewer/paper_reviewer_matching)|[paper_reviewer_matching](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/paper_reviewer_matching), [reviewers](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/reviewers)|
|PRX|Feeds-1|N|MAP|[feeds_1](https://huggingface.co/datasets/allenai/scirepeval/viewer/feeds_1)|[feeds_1](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/feeds_1)|
|PRX|Feeds-M|N|MAP|[feeds_m](https://huggingface.co/datasets/allenai/scirepeval/viewer/feeds_m)|[feeds_m](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/feeds_m)|
|PRX|[SciDocs-Cite](https://github.com/allenai/scidocs)|N|MAP, NDCG|[scidocs_view_cite_read](https://huggingface.co/datasets/allenai/scirepeval/viewer/scidocs_view_cite_read)|[scidocs_cite](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/scidocs_cite)|
|PRX|[SciDocs-CoCite](https://github.com/allenai/scidocs)|N|MAP, NDCG|[scidocs_view_cite_read](https://huggingface.co/datasets/allenai/scirepeval/viewer/scidocs_view_cite_read)|[scidocs_cocite](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/scidocs_cocite)|
|PRX|[SciDocs-CoView](https://github.com/allenai/scidocs)|N|MAP, NDCG|[scidocs_view_cite_read](https://huggingface.co/datasets/allenai/scirepeval/viewer/scidocs_view_cite_read)|[scidocs_view](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/scidocs_view)|
|PRX|[SciDocs-CoRead](https://github.com/allenai/scidocs)|N|MAP, NDCG|[scidocs_view_cite_read](https://huggingface.co/datasets/allenai/scirepeval/viewer/scidocs_view_cite_read)|[scidocs_read](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/scidocs_read)|
|SRCH|Search|Y|NDCG|[search](https://huggingface.co/datasets/allenai/scirepeval/viewer/search)|[search](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/search)|
|SRCH|Feeds-Title|N|MAP|[feeds_title](https://huggingface.co/datasets/allenai/scirepeval/viewer/feeds_title)|[feeds_title](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/feeds_title)|
|SRCH|[TREC-CoVID](https://ir.nist.gov/trec-covid/data.html)|N|NDCG|[trec_covid](https://huggingface.co/datasets/allenai/scirepeval/viewer/trec_covid)|[trec_covid](https://huggingface.co/datasets/allenai/scirepeval_test/viewer/trec_covid)|

*S2AND requires the evaluation dataset in a specific format so to evaluate your model on the task please follow [these](https://github.com/allenai/scirepeval/blob/main/BENCHMARKING.md#s2and) instructions.

**Combinations of multiple datasets - [1](https://mimno.infosci.cornell.edu/data/nips_reviewer_data.tar.gz), [2](https://web.archive.org/web/20211015210300/http://sifaka.cs.uiuc.edu/ir/data/review.html), [3](https://ieee-dataport.org/open-access/retrorevmatchevalicip16-retrospective-reviewer-matching-dataset-and-evaluation-ieee-icip), also dataset of papers authored by potential reviewers is required for evaluation; hence the multiple dataset configs.

## License
The aggregate benchmark is released under [ODC-BY](https://opendatacommons.org/licenses/by/1.0/) license. By downloading this data you acknowledge that you have read and agreed to all the terms in this license.
For constituent datasets, also go through the individual licensing requirements, as applicable. 

