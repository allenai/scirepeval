# a = [1, 4, 7]
#
# import itertools
#
#
# def identity(line):
#     return [("aps", [line, line, line]), ("aps", [line, line + 1, line + 1])]
#
#
# l = itertools.chain(*map(identity, a))
# print(list(l))


# import json
#
# with open("sample_data/mesh_small.json") as f:
#     # [{"corpus_id":...,"title":..., "abstract":...}...]
#     task_data = []
#
#     task_data = json.load(f)
#
# print(len(task_data))


# from sklearn.preprocessing import MultiLabelBinarizer
#
# mlb = MultiLabelBinarizer()
# mlb.fit([0, 1,2])
# print(mlb.classes_)
# import random
# from random import choice
#
# random.seed(42)
# for i in range(4):
#     print(choice([1,2,3,4,5]))


# a = iter([_ for _ in range(500)])
#
# batch = []
#
# try:
#     while True:
#         for i,d in enumerate(a):
#             if i < 100:
#                 batch.append((i,next(a)))
#         for x in batch:
#             print(x)
#
#         for i,d in enumerate(a):
#                 batch.append((i,next(a)))
#         batch = []
# except StopIteration:
#     print(batch)

# from pl_training import PhantasmLight
#
# model = PhantasmLight.load_from_checkpoint()
#
# model.encoder.save_pretrained("./lightning_logs/trial/mixed_prop/checkpoints/model/")
# model.tokenizer.save_pretrained("./lightning_logs/trial/mixed_prop/checkpoints/tokenizer/")
# model.tokenizer.save_vocabulary("./lightning_logs/trial/mixed_prop/checkpoints/tokenizer/")

# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
# print(tokenizer.sep_token)

# import torch
#
# A = torch.tensor([22,  2,  3, 22,  8,  3,  1,  7, 29, 26,  5, 15, 28, 18, 11,  5])
# B = torch.nn.functional.one_hot(A, num_classes=30).T
# C = B[A]
# print(C)
# C.fill_diagonal_(0)
# print(C)

# import torch
#
# x = torch.tensor([
#     [0, 2, 0, 1], # A
#     [1, 3, 0, 1], # B
#     [1, 2, 0, 1], # C
#     [0, 3, 0, 1], # D
#     [1, 2, 0, 1], # E
#     [1, 3, 0, 1]  # F
# ])
#
# y = [(1,"q"), (2,"c"), (3, "c"), (4, "q"), (5, "c"), (6, "c")]
#
# q_idx = torch.tensor([i for i,d in enumerate(y) if d[1]=="q"])
# c_idx = torch.tensor([i for i,d in enumerate(y) if d[1]=="c"])
# print(x[q_idx])

# print(q_idx)
# print(c_idx)

#
# y = torch.sort(torch.unique(x[:,1])).values
#
# print(x[(x[:,1]==y[1]).nonzero(as_tuple=True)])


from transformers import BertModelWithHeads
import torch
# model = BertModelWithHeads.from_pretrained(
#     "bert-base-uncased"
# )
# embeddings = torch.mean(model.bert.embeddings.word_embeddings.weight, dim=0).reshape(1, -1)
# b = torch.nn.functional.normalize(embeddings, p=2.0, dim= 1)
# a = torch.rand(5,768)
# print(a)
# print(a[-2:,:] + b)
A = torch.tensor([1,2,3])
B = torch.tensor([4,5,6])
D = torch.tensor([4,5,6])
C = torch.stack([A,B, D], dim=1)
#D = torch.tensor([7,8,9])
print(C)
# from transformers.adapters.composition import Fuse
#
# # Load the pre-trained adapters we want to fuse
# model.load_adapter("nli/multinli@ukp", load_as="multinli", with_head=False)
# model.load_adapter("sts/qqp@ukp", with_head=False)
# model.load_adapter("nli/qnli@ukp", with_head=False)
# # Add a fusion layer for all loaded adapters
# model.add_adapter_fusion(Fuse(*["multinli", "qqp", "qnli"]))
# model.set_active_adapters(Fuse(*["multinli", "qqp", "qnli"]))
