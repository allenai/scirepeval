from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer

instr_format = "Represent the Scientific documents for "


class InstructorModel:
    def __init__(self, embed_model: str):
        self.encoder = INSTRUCTOR(embed_model)
        self.task_id = None
        self.instruction_map = {"[CLF]": f"{instr_format} classification: ", "[RGN]": f"{instr_format} regression: ",
                                "[PRX]": f"{instr_format} retrieving similar similar documents: ",
                                "[SRCH]": {"q": "Represent the Scientific query for retrieving relevant documents: ",
                                           "c": f"{instr_format} for retrieval: "}}
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.tokenizer.sep_token = self.tokenizer.eos_token

    def __call__(self, batch, batch_ids=None):
        if type(self.task_id) != dict:
            batch = [[self.instruction_map[self.task_id], b] for b in batch]
        else:
            instructions = [f"{self.instruction_map['SRCH'][b[1]]}{batch[i]}" for i, b in enumerate(batch_ids)]
            batch = [[ins, b] for ins, b in zip(instructions, batch)]
        batch_embed = self.encoder.encode(batch, convert_to_numpy=False, convert_to_tensor=True, device="cuda")
        return batch_embed
