## Benchmarking
We provide our trained models on the HuggingFace models [hub](https://huggingface.co/models?search=scirepeval) to replicate the results in Table 2 from the paper.

|Model|In-Train|Out-of-Train|SciDocs|Average|
|--|--|--|--|--|
|[SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased)|51.2|52.6|69.0|58.1|
|[SPECTER](https://huggingface.co/allenai/specter)|54.3|57.4|89.1|67.9|
|[SciNCL](https://huggingface.co/malteos/scincl)|55.3|57.8|**90.8**|69.0|
|SciNCL + MTL CLS|59.8|56.7|89.6|69.2|
|[SciNCL + MTL CTRL](https://huggingface.co/allenai/scirepeval_ctrl)|61.6|57.7|89.9|70.2|
|[SciNCL PALs](https://huggingface.co/allenai/scirepeval_pals)|61.9|58.3|90.0|70.5|
|SciNCL Adapters ([CLF](https://huggingface.co/allenai/scirepeval_adapters_clf), [QRY](https://huggingface.co/allenai/scirepeval_adapters_qry), [RGN](https://huggingface.co/allenai/scirepeval_adapters_rgn), [PRX](https://huggingface.co/allenai/scirepeval_adapters_prx))|61.4|58.8|90.3|70.7|
|[SciNCL Adapters Fusion](https://us-east-1.console.aws.amazon.com/s3/buckets/ai2-s2-research-public?region=us-west-2&prefix=scirepeval/adapters/&showversions=false)|61.5|**58.8**|89.9|70.5|
|SciNCL Adapters + MTL CTRL|62.2|**58.8**|**90.7**|**71.0**|


We provide a test script - [scirepeval.py](https://github.com/allenai/scirepeval/blob/main/scirepeval.py) to evaluate one of the above models or a custom trained model on all the tasks in the benchmark.
The tasks can be configured as required in [scirepeval_tasks.jsonl](https://github.com/allenai/scirepeval/blob/main/scirepeval_tasks.jsonl).

The following are used as task ids in the code and serve as either control codes or module identifiers for each task type:

``TASK_IDS = {"classification": "[CLF]", "regression": "[RGN]", "proximity": "[PRX]",
            "adhoc_search": {"query": "[QRY]", "candidates": "[PRX]"}}``

Execute one of the following commands to evaluate a model on SciRepEval:
<a name="models"></a>

**Base/MTL CLS**
```bash
python scirepeval.py -m allenai/specter
```
**MTL CTRL**
```bash
python scirepeval.py -m allenai/scirepeval_ctrl --ctrl-tokens
```
**PALs**
```bash
python scirepeval.py --mtype pals -m allenai/scirepeval_pals
```
**Adapters**
```bash
python scirepeval.py --mtype adapters -m malteos/scincl --adapters-dir <local checkpoint directory with adapter module weights>
									OR
python scirepeval.py --mtype adapters -m malteos/scincl --adapters-chkpt '{"[CLF]": "allenai/scirepeval_adapters_clf", "[QRY]": "allenai/scirepeval_adapters_qry", "[RGN]": "allenai/scirepeval_adapters_rgn", "[PRX]": "allenai/scirepeval_adapters_prx"}'
```

**Fusion**
```bash
python scirepeval.py --mtype fusion -m <huggingface base model name/local checkpoint path> --adapters-dir <local checkpoint directory with fusion module weights>
```

The script generates embeddings and evaluates on each task as per the metric mentioned in the paper. By default the result report is created in `<ROOT>/scirepeval_results.json`

### Sample Report
```json
{
    "Biomimicry": {
        "complete": {
            "f1": 71.18
        },
        "few_shot": [
            {
                "sample_size": 64,
                "results": {
                    "f1": 38.514
                }
            },
            {
                "sample_size": 16,
                "results": {
                    "f1": 22.3444
                }
            }
        ]
    },
    "DRSM": {
        "complete": {
            "f1_macro": 76.36
        },
        "few_shot": [
            {
                "sample_size": 64,
                "results": {
                    "f1_macro": 61.842000000000006
                }
            },
            {
                "sample_size": 24,
                "results": {
                    "f1_macro": 53.21420000000001
                }
            }
        ]
    },
    "Feeds-1": {
        "map": 81.03
    },
    "Feeds Title": {
        "map": 78.85
    }
}
```

<a name="s2and"></a>
### S2AND evaluation
S2AND evaluation requires the data to be cached locally in a specific format. We provide a helper script to generate the document representations for S2AND before evaluating them.

**Step 1**

Obtain the data from AWS S3:
```bash
mkdir s2and && cd s2and
aws s3 --no-sign-request sync s3://ai2-s2-research-public/scirepeval/test/s2and .
```
**Step 2** 

Generate Embeddings for all the paper blocks. The various model parameters are same as scirepeval.py, provide those to initialize the required model type.
```bash
python s2and_embeddings.py --mtype <model type> -m <model checkpoint> --adapters-dir <adapters dir or chkpt> --data-dir <path to S2AND data> --suffix <suffix for embedding file name>
```
**Step 3**

Run S2AND evaluation.
Setup S2AND as in [repo](https://github.com/allenai/S2AND) and change the configuration to point to your data location.

Run the following command:
```bash
cd S2AND; python scripts/custom_block_transfer_experiment_seed_paper.py --custom_block_path <data>/blocks --experiment_name mini_customblock_phantasm_v1 --exclude_medline --emb_suffix _<suffix>.pkl
```
### Filtering Tasks
#### By Name
```python
from scirepeval import SciRepEval
from evaluation.encoders import Model

#Base/MTL CLS
model = Model(variant="default", base_checkpoint="allenai/specter")

#MTL CTRL
model = Model(variant="default", base_checkpoint="allenai/scirepeval_ctrl", use_ctrl_codes=True)

#PALs
model = Model(variant="pals", base_checkpoint="allenai/scirepeval_pals", all_tasks=["[CLF]", "[QRY]", "[RGN]", "[PRX]"])

#Adapters/Fusion
adapters_dict = {"[CLF]": "allenai/scirepeval_adapters_clf", "[QRY]": "allenai/scirepeval_adapters_qry", "[RGN]": "allenai/scirepeval_adapters_rgn", "[PRX]": "allenai/scirepeval_prx"}
model = Model(variant=<"adapters"|"fusion">, base_checkpoint="malteos/scincl", adapters_load_from=adapters_dict, all_tasks=["[CLF]", "[QRY]", "[RGN]", "[PRX]"])

#Choose the task names from scirepeval_tasks.jsonl
evaluator = SciRepEval(task_list=["Biomimicry", "DRSM", "TREC-CoVID", "Feeds-1"])
evaluator.evaluate(model, "scirepeval_results.json") 
```

#### By Task Type
```python
from scirepeval import SciRepEval
from evaluation.encoders import Model

#Create a model instance as in previous example
model = Model(variant="default", base_checkpoint="allenai/specter")

#Choose the task types from (classification, regression, proximity and adhoc_search)
evaluator = SciRepEval(task_formats=["classification", "regression"])
evaluator.evaluate(model, "scirepeval_results.json") 
```


