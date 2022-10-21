import sys

sys.path.append('../')

from evaluation.encoders import Model
from evaluation.few_shot_evaluator import FewShotEvaluator, SupervisedTask

# model = Model(base_checkpoint="allenai/specter")
model = Model(base_checkpoint="../lightning_logs/full_run/scincl_ctrl/checkpoints/", task_id="[CLF]",
              use_ctrl_codes=True)
# model = Model(base_checkpoint="malteos/scincl", variant="adapters",
#               adapters_load_from="../lightning_logs/full_run/scincl_adapters/checkpoints/", task_id="[CLF]")
evaluator = FewShotEvaluator("drsm", SupervisedTask.CLASSIFICATION, ("allenai/scirepeval", "drsm"),
                             ("allenai/scirepeval_test", "biomimicry"), model=model, metrics=("f1_macro",),
                             sample_size=16, num_iterations=50)

embeddings = evaluator.generate_embeddings()

evaluator.evaluate(embeddings)
