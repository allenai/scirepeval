## Evaluation

 - SciRepEval can be used to evaluate scientific document representations on 4 task types - classification, regression, proximity based retrieval ( a document is the query) and ad-hoc search ( raw text query).
 - The evaluation process for each task consists of 2 steps - representation generation with a model and raw metadata; and evaluating these representation as features of the labelled test examples using a suitable metric.

To reproduce the results in the paper for all or a collection of tasks in SciRepEval, follow the steps in [BENCHMARKING.md](https://github.com/allenai/scirepeval/blob/main/BENCHMARKING.md).


### Custom Evaluation
#### SciRepEval config
The evaluation setup for the existing tasks in SciRepEval can be configured in [scirepeval_tasks.jsonl](https://github.com/allenai/scirepeval/blob/main/scirepeval_tasks.jsonl).
These config parameters are internally parsed by the evaluators to generate the document representations and compute the relevant metric.

**Example task config**:
```json
{
    "name": "Biomimicry",
    "type": "classification",
    "data":
    {
        "meta":
        {
            "name": "allenai/scirepeval",
            "config": "biomimicry"
        },
        "test":
        {
            "name": "allenai/scirepeval_test",
            "config": "biomimicry"
        }
    },
    "metrics":
    [
        "f1"
    ],
    "few_shot":
    [
        {
            "sample_size": 64,
            "iterations": 50
        },
        {
            "sample_size": 16,
            "iterations": 100
        }
    ]
}
```
**Notes**

 1. `"name"` - identifier for the task, can be utilized when filtering the tasks for evaluation.
 2. `"type"`- can be one of `{"classification", "regression", "proximity", "adhoc_search"}`, for multi-label classification, provide additional `"multi_label"=true` flag.
 3. `"data"` is required and expects at-least two entries: `"meta"` for the raw test data with title and abstracts for representation generation and `"test"` for the labelled examples. These can be local file paths or HuggingFace datasets. 
 4. `"metrics"` is a list of the metrics to be computed for the task. These can be customized based on task type as follows:
 ```python
 if "type" == "classification":
	 metrics can be {"f1", "accuracy", "precision", "recall", "{f1|precision|recall}_{macro|micro}"}
elif "type" == "regression":
	metrics can be {"mse", "r2", "pearsonr","kendalltau"}
else:
	metrics can be anything allowed in pytrec_eval* 
 ``` 
 *[pytrec_eval](https://github.com/cvangysel/pytrec_eval)
 
 5. Classification tasks can be additionally evaluated in few shot mode, provide a list of `"sample_size"` and `"iterations"`.
 6. To avoid generating embeddings in every run, these can be cached and re-loaded in future runs by providing the `"embedding"` config as-
 ```json
 "embeddings":{"save":"<embeddings_dir>/<embeddings_file>.jsonl"} OR "embeddings":{"load":<embeddings_dir>/<embeddings_file>.jsonl}
 ``` 

#### Custom Tasks
For evaluating on new tasks from any of the four task types in SciRepEval, create the task config json as above and either append it to **scirepeval_tasks.jsonl** or add it to a new config file.

To evaluate on all tasks: 
Select model parameters as in [here](https://github.com/allenai/scirepeval/blob/main/BENCHMARKING.md#models).
```bash
python scirepeval.py --m <huggingface model name/local checkpoint path> --tasks-confg <tasks config file path> --output <output path>
```
OR

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
evaluator = SciRepEval(tasks_config=<task config path>, task_list:Optional=[...], task_format:Optional=[...])
evaluator.evaluate(model, <output jsonl path>) 
```


