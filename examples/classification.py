import sys

sys.path.append('../')

from evaluation.encoders import Model
from evaluation.evaluator import SupervisedEvaluator, SupervisedTask

#default no control codes
model = Model(base_checkpoint="allenai/specter")

#default control codes
# model = Model(base_checkpoint="../lightning_logs/full_run/scincl_ctrl/checkpoints/", task_id="[CLF]", use_ctrl_codes=True)

#single adapters
# model = Model(base_checkpoint="malteos/scincl", variant="adapters",
#               adapters_load_from="../lightning_logs/full_run/scincl_adapters/checkpoints/", task_id="[CLF]")


evaluator = SupervisedEvaluator("biomimicry", SupervisedTask.CLASSIFICATION, ("allenai/scirepeval", "biomimicry"),
                                ("allenai/scirepeval_test", "biomimicry"), model, metrics=("f1",))

embeddings = evaluator.generate_embeddings()

evaluator.evaluate(embeddings)
