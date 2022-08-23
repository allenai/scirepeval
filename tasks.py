from typing import Dict

from torch import nn
import torch
import torch.nn.functional as F
import json


class TaskFamily:
    def __init__(self, name, loss, data_files, type, multi_label=False, input_fields=None,
                 labels_field=None, labels=None, ctrl_token=None, head=None, contrastive_loss=None):
        if input_fields is None:
            input_fields = ["title", "abstract"]
        self.name = name
        self.data_files = data_files
        self.type = type
        self.multi_label = multi_label
        self.loss = loss
        self.contrastive_loss = contrastive_loss
        self.ctrl_token = ctrl_token
        self.head = head
        self.labels = labels
        self.labels_field = labels_field
        self.input_fields = input_fields

    def __str__(self):
        obj_dict = self.__dict__.copy()
        del_fields = ["head", "loss"]
        for field in del_fields:
            del obj_dict[field]
        return json.dumps(obj_dict)


class TaskHead(nn.Module):
    def __init__(self, num_labels, dim=768):
        super().__init__()
        self.dim = dim
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(dim, num_labels)

    def forward(self, encoding):
        return self.linear(self.dropout(encoding))


class SCLLoss(nn.Module):
    def __init__(self, temp=0.3):
        super(SCLLoss, self).__init__()
        self.temp = temp

    def forward(self, encoding, y, num_classes):
        norm_encoding = F.normalize(encoding, p=2, dim=1)
        dot_prod = torch.matmul(norm_encoding, norm_encoding.T) / self.temp
        y_mask = torch.nn.functional.one_hot(y, num_classes=num_classes).T[y]
        diag_mask = torch.ones(y_mask.shape, device=torch.device('cuda')).fill_diagonal_(0)
        den = torch.exp(dot_prod) * diag_mask
        inner = dot_prod - torch.log(torch.sum(den, dim=1) + 1e-10)
        con_loss = inner * y_mask * diag_mask
        scl_loss = -1.0 * torch.sum(con_loss, dim=1) / (torch.sum(y_mask, dim=1))
        return scl_loss


# Triplet will be an object of TaskFamily as no separate head needed
class TripletLoss(nn.Module):
    """
    Triplet loss: copied from  https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/model.py#L159 without any change
    """

    def __init__(self, margin=1.0, distance='l2-norm', reduction='mean'):
        """
        Args:
            margin: margin (float, optional): Default: `1`.
            distance: can be `l2-norm` or `cosine`, or `dot`
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: 'mean'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        self.reduction = reduction

    def forward(self, query, positive, negative):
        if self.distance == 'l2-norm':
            distance_positive = F.pairwise_distance(query, positive)
            distance_negative = F.pairwise_distance(query, negative)
            losses = F.relu(distance_positive - distance_negative + self.margin)
        elif self.distance == 'cosine':  # independent of length
            distance_positive = F.cosine_similarity(query, positive)
            distance_negative = F.cosine_similarity(query, negative)
            losses = F.relu(-distance_positive + distance_negative + self.margin)
        elif self.distance == 'dot':  # takes into account the length of vectors
            shapes = query.shape
            # batch dot product
            distance_positive = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                positive.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0], )
            distance_negative = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                negative.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0], )
            losses = F.relu(-distance_positive + distance_negative + self.margin)
        else:
            raise TypeError(f"Unrecognized option for `distance`:{self.distance}")

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses
        else:
            raise TypeError(f"Unrecognized option for `reduction`:{self.reduction}")


def load_tasks(tasks_config_file: str = "sample_data/tasks_config.json") -> Dict[str, TaskFamily]:
    def load_labels(labels_file: str) -> Dict[str, str]:
        with open(labels_file, "r") as f:
            labels = f.readlines()
        labels = {l.strip(): i for i, l in enumerate(labels)}
        return labels

    import json
    task_dict = dict()
    task_config = json.load(open(tasks_config_file, "r"))
    for task in task_config:
        if task["type"] == "classification":
            task["labels"] = load_labels(task["labels"])
            task["head"] = TaskHead(num_labels=len(task["labels"]))
            if task.get("multi_label"):
                task["loss"] = nn.BCEWithLogitsLoss(reduction="none")
            else:
                task["loss"] = nn.CrossEntropyLoss(reduction="none")
                use_contrastive = task.pop("contrastive", False)
                if use_contrastive:
                    task["contrastive_loss"] = SCLLoss()
        elif task["type"] == "regression":
            task["head"] = TaskHead(num_labels=1)
            task["loss"] = nn.MSELoss(reduction="none")
        else:
            task["loss"] = TripletLoss(reduction="none")
        task_dict[task["name"]] = TaskFamily(**task)
    return task_dict
