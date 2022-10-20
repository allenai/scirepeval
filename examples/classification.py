import sys

sys.path.append('../')

from evaluation.encoders import Model
from evaluation.evaluator import SupervisedEvaluator, SupervisedTask

model = Model(base_checkpoint="allenai/specter")
evaluator = SupervisedEvaluator(SupervisedTask.CLASSIFICATION, ("allenai/scirepeval", "biomimicry"),
                                ("allenai/scirepeval_test", "biomimicry"), model, metrics=("binary",))

embeddings = evaluator.generate_embeddings()

evaluator.evaluate(embeddings)
