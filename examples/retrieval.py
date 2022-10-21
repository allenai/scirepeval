import sys

sys.path.append('../')

from evaluation.encoders import Model
from evaluation.evaluator import IREvaluator
from reviewer_matching import ReviewerMatchingEvaluator
# model = Model(base_checkpoint="allenai/specter")
model = Model(base_checkpoint="../lightning_logs/full_run/scincl_ctrl/checkpoints/",
              task_id={"query": "[QRY]", "candidate": "[SAL]"},
              use_ctrl_codes=True)
# model = Model(base_checkpoint="malteos/scincl", variant="adapters",
#               adapters_load_from="../lightning_logs/full_run/scincl_adapters/checkpoints/", task_id="[CLF]")
# evaluator = IREvaluator("feeds_1", ("allenai/scirepeval", "feeds_1"), ("allenai/scirepeval_test", "feeds_1"), model,
#                         metrics=("map", "ndcg",))
#
# embeddings = evaluator.generate_embeddings()
#
# evaluator.evaluate(embeddings)

# evaluator = IREvaluator("feeds_1", ("allenai/scirepeval", "feeds_title"), ("allenai/scirepeval_test", "feeds_title"),
#                         model, metrics=("map", "ndcg",))
evaluator = ReviewerMatchingEvaluator("paper reviewer evaluation", ("allenai/scirepeval", "paper_reviewer_matching"), ("allenai/scirepeval_test", "paper_reviewer_matching"),
                         ("allenai/scirepeval_test", "reviewers"), model, metrics=("map", "ndcg",))

embeddings = evaluator.generate_embeddings()

evaluator.evaluate(embeddings)
