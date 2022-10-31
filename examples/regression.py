import sys

sys.path.append('../')

from evaluation.encoders import Model
from evaluation.evaluator import SupervisedEvaluator, SupervisedTask

model = Model(base_checkpoint="allenai/specter")
# model = Model(base_checkpoint="../lightning_logs/full_run/scincl_ctrl/checkpoints/", task_id="[RGN]", use_ctrl_codes=True)
# model = Model(base_checkpoint="malteos/scincl", variant="adapters",
#               adapters_load_from="../lightning_logs/full_run/scincl_adapters/checkpoints/", task_id="[CLF]")
evaluator = SupervisedEvaluator("max hIndex", SupervisedTask.REGRESSION, ("allenai/scirepeval", "peer_review_score_hIndex"),
                                ("allenai/scirepeval_test", "hIndex"), model, metrics=("pearsonr","kendalltau"))

embeddings = evaluator.generate_embeddings()

evaluator.evaluate(embeddings)
