## Training
The code available as part of this sub-directory can be used to train a general purpose multi-task model or the multi-format based models introduced in [SciRepEval](https://openreview.net/pdf?id=zfiYcbeQkH).

Post the quick setup step in ReadMe, you can choose to train the following base models:
(Parenthesis denote how they are referred in the paper)
 1. General multi task model (MTL CLS) - \[CLS\] token embedding is considered document representation
 2. Multi-task training w. Control Codes (MTL CTRL) - Control codes prepended to input and their embedding is considered document representation
 3. [BERT PALs](https://github.com/AsaCooperStickland/Bert-n-Pals) (PALs) - Task specific modules
 4. [Adapters and Fusion](https://github.com/adapter-hub/adapter-transformers) - Task specific adapters

#### Step 1
Define the tasks and associated metadata in a json config file. Refer to [sample_data/tasks_config.json](https://github.com/allenai/scirepeval/blob/main/training/sample_data/tasks_config.json) for SciRepEval training config.
*Example config:*
```json
{
    "name": "fos",
    "type": "classification",
    "multi_label": true,
    "data_files":
    {
        "train": "../../scirepeval_data/train/fos/train.jsonl",
        "dev": "../../scirepeval_data/train/fos/val.jsonl"
    },
    "labels": "sample_data/fos_labels.txt",
    "labels_field": "labels_text",
    "ctrl_token": "[CLF]",
    "sample_size":
    {
        "train": 600000,
        "dev": 40000
    }
}
```
**Note**

 - `"type"` can be one of `["classification", "regression", "ir", "triplet"]`.
 - `"classification"` is suitable for tasks with categorical (discrete) labels,;`"regression"` for tasks with continuous labels; `"ir"` for retrieval tasks formatted as `{"query": X, "candidates": [{}]}` and `"triplet"` for contrastive learning tasks formatted as `{"query": q, "pos": p, "neg": n}`.
 - For multi label classification, add  `"multi_label": true` as in the above example.
 - By default the pre-processing code expects "title" and "abstract" in every example. To process specific fields, provide  additional property as `"input_fields": ["title", "abstract", "venue", "year"]`.
 - For models apart from MTL CLS, provide the `"ctrl_token"` associated with each task, for MTL CTRL it works as the special control code and for PALs and Adapters it acts as the task id to determine the module to be used in the forward pass.
 - Some "ir" tasks like ad-hoc search \[SRCH\] might require different control codes forthe query and candidates which can be provided as `"ctrl_token": {"query": "[QRY]", "candidates": "[PRX]"}`. For PALs and Adapters, this task id is internally resolved to feed the queries and candidates to their relevant modules.
 - `"sample_size"` is not required if all the samples are to be processed for the splits.
 - If loading data from Huggingface datsets, instead of `"data_files"`, you can provide parameters for `load_dataset` method as - `"dataset": {"path": <hf dataset name>, "name": <optional config name for dataset with multiple configs>}`.
 - ``if "type"=="regresion": <provide the "labels_field"> elif "type" =="classification": <provide the "labels" and "labels_field"> ``
 - Losses associated with each task type:
 
|Type|Loss |
|--|--|
| Classification |Cross Entropy |
|Multi-label Classification|Binary Cross Entropy|
|Regression|Mean Squared Error|
|IR/Triplet|Triplet or Contrastive Loss|


#### Step 2
To run the training script with default params, based upon the type of models you want to train run one of the following commands:
**MTL CLS**
```bash
python pl_training.py --gpu 2 <base model name/chkpoint path> <tokenizer name/chkpoint path> <expt name>
```

**MTL CTRL**
```bash
python pl_training.py --gpu 2 --ctrl-tokens <base model name/chkpoint path> <tokenizer name/chkpoint path> <expt name>
```

**PALs**

Requires pals config file for additional model configuration. Files present under `bert_pals_config` directory.
```bash
python pl_training.py --gpu 2 --pals-config pals.config.json <base model name/chkpoint path> <tokenizer name/chkpoint path> <expt name>
```
**Adapters**
```bash
python pl_training.py --gpu 2 --ctrl-tokens --adapter-type single <base model name/chkpoint path> <tokenizer name/chkpoint path> <expt name>
```
**Fusion**

    python pl_training.py --gpu 2 --ctrl-tokens --adapter-type fusion <base model name/chkpoint path> <tokenizer name/chkpoint path> <expt name>

### Additional Parameters

```positional arguments:

model HuggingFace model to be used

tokenizer HuggingFace tokenizer to be used

version experiment version

  

optional arguments:

-h, --help  show this help message and exit

--tasks-confg TASKS_CONFG path to the task config file

--output OUTPUT dir to save checkpoints and finetuned model

--pals-config PALS_CONFIG name of config file for PALS architecture

--adapter-type ADAPTER_TYPE type of adapter architecture (single/fusion)

--batch-size BATCH_SIZE batch size

--lr LR initial learning rate

--peak-lr PEAK_LR initial learning rate

--warmup WARMUP number of warmup steps

--epochs EPOCHS number of epochs

--grad-accum GRAD_ACCUM grad accumulation steps

--ctrl-tokens use control codes for tasks

--gpu GPU number of gpus

--max_len MAX_LEN max sequence length

--val_check_interval VAL_CHECK_INTERVAL validation loop interval

--checkpoint CHECKPOINT resume from checkpoint path
```

TensorBoard logs and checkpoints are written to `<output>/<version>` directory.
