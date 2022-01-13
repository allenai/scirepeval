from torch import nn


class TaskFamily:
    def __init__(self, name, loss, dataset, ctrl_token="[CLS]"):
        self.name = name
        self.loss = loss
        self.dataset = dataset
        self.ctrl_token = ctrl_token


# BCE for FoS, CE for MeSH
class Classification(TaskFamily):
    def __init__(self, name, num_labels, loss, dataset, dim=768, ctrl_token="[CLS]"):
        super().__init__(name, loss, dataset, ctrl_token)
        self.head = nn.Linear(dim, num_labels)
        self.head.weight.data.normal_(mean=0.0, std=0.02)
        if self.head.bias is not None:
            self.head.bias.data.zero_()
# Triplet will be an object of TaskFamily as no separate head needed
