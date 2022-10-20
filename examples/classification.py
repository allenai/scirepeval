import sys

sys.path.append('../')

from evaluation.encoders import Model
from evaluation.evaluator import SupervisedEvaluator, SupervisedTask

#model = Model(base_checkpoint="allenai/specter")
model = Model(base_checkpoint="../lightning_logs/full_run/scincl_ctrl/checkpoints/", task_id="[CLF]", use_ctrl_codes=True)
evaluator = SupervisedEvaluator(SupervisedTask.CLASSIFICATION, ("allenai/scirepeval", "biomimicry"),
                                ("allenai/scirepeval_test", "biomimicry"), model, metrics=("binary",))

embeddings = evaluator.generate_embeddings()

evaluator.evaluate(embeddings)
