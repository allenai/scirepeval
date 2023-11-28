from scirepeval import SciRepEval
from evaluation.encoders import Model
from mdcr import MDCREvaluator
import sys
import json

args=sys.argv
#MTL CTRL
model1 = Model(variant="default", task_id="[PRX]", base_checkpoint=args[1], use_ctrl_codes=True, use_fp16=False)

#Adapters/Fusion
# adapters_dict = {"[CLF]": "allenai/scirepeval_adapters_clf", "[QRY]": "allenai/scirepeval_adapters_qry", "[RGN]": "allenai/scirepeval_adapters_rgn", "[PRX]": "allenai/scirepeval_adapters_prx"}
model2 = Model(variant="adapters", task_id="[PRX]", base_checkpoint="allenai/specter", adapters_load_from=args[2], all_tasks=["[CLF]", "[QRY]", "[RGN]", "[PRX]"])

models = [model2, model1]


# evaluator = MDCREvaluator("mdcr", "../mdcr/mdcr_test_data.jsonl", "../mdcr/mdcr_test.json", models, batch_size=32)
# embeddings = evaluator.generate_embeddings(save_path="mdcr_embeddings.json")
# evaluator.evaluate(embeddings, args[3])

#Choose the task names from scirepeval_tasks.jsonl
tasks, config = [], "scirepeval_tasks.jsonl"
if len(args) > 4:
    config = args[4]
if len(args) > 6:
    tasks = args[6:]
if not tasks:
    evaluator = SciRepEval(tasks_config=config,batch_size=16)
else:
    evaluator = SciRepEval(tasks_config=config, batch_size=16, task_list=tasks)
print(config)
evaluator.evaluate(models, args[3])


