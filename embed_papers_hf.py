# simple script for embedding papers using huggingface Specter
# requirement: pip install --upgrade transformers==4.2.2
from transformers import AutoModel, AutoTokenizer
import json
import argparse
from tqdm.auto import tqdm
import pathlib
import torch
import decimal
import hashlib


class Dataset:

    def __init__(self, data_path, model_dir, max_length=512, batch_size=32, ctrl_token=None,
                 fields=["title", "abstract"]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir + "/tokenizer/")
        self.max_length = max_length
        self.batch_size = batch_size
        self.ctrl_token = ctrl_token
        self.fields = fields
        # data is assumed to be a json file
        # [{"corpus_id":...,"title":..., "abstract":...}...]
        try:
            self.data = json.load(open(data_path, "r"))
        except:
            with open(data_path) as f:
                self.data = [json.loads(line) for line in f]
        print("Special tokens:")
        print(self.tokenizer.all_special_tokens)

    def __len__(self):
        return len(self.data)

    def batches(self):
        # create batches
        batch = []
        batch_ids = []
        batch_size = self.batch_size
        i = 0
        for d in self.data:
            if "corpus_id" in d:
                text = "" if not self.ctrl_token else self.ctrl_token + " "
                text += d['title'] + ' '
                if d.get('abstract'):
                    text += f'{self.tokenizer.sep_token} ' + d.get('abstract')
                if (i) % batch_size != 0 or i == 0:
                    batch_ids.append(d["corpus_id"])
                    batch.append(text)
                else:
                    input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                               return_tensors="pt", max_length=self.max_length)
                    yield input_ids.to('cuda'), batch_ids
                    batch_ids = [d["corpus_id"]]
                    batch = [text]
                i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt", max_length=self.max_length)
            input_ids = input_ids.to('cuda')
            yield input_ids, batch_ids


class QueryDataset(Dataset):
    def __len__(self):
        return 11 * len(self.data)

    def batches(self):
        # create batches
        batch = []
        batch_ids = []
        batch_size = self.batch_size
        i = 0
        for d in self.data:
            d_pair = [d["query"]] + d["candidates"]
            for dp in d_pair:
                key = ""
                text = []
                if type(dp) == dict:
                    if "corpus_id" in dp:
                        key = dp["corpus_id"]
                        for field in self.fields:
                            if dp[field]:
                                if type(dp[field]) == float:
                                    dp[field] = str(int(dp[field]))
                                text.append(dp[field])
                        text = (f" {self.tokenizer.sep_token} ".join(text)).strip()
                else:
                    hash_object = hashlib.md5(str(dp).encode("utf-8"))
                    key = hash_object.hexdigest()
                    text = dp
                if key:
                    if self.ctrl_token:
                        text = self.ctrl_token + " " + text
                    if (i) % batch_size != 0 or i == 0:
                        batch_ids.append(key)
                        batch.append(text)
                    else:
                        input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                                   return_tensors="pt", max_length=self.max_length)
                        yield input_ids.to('cuda'), batch_ids
                        batch_ids = [key]
                        batch = [text]
                    i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt", max_length=self.max_length)
            input_ids = input_ids.to('cuda')
            yield input_ids, batch_ids


class Model:

    def __init__(self, token_idx, model_dir):
        self.model = AutoModel.from_pretrained(model_dir + "/model/")
        self.model.to('cuda')
        self.model.eval()
        self.token_idx = token_idx

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        return output.last_hidden_state[:, self.token_idx, :]  # cls token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='path to a json file containing paper metadata')
    parser.add_argument('--model-dir', help='path to the HF model and tokenizer')
    parser.add_argument('--output', help='path to write the output embeddings file. '
                                         'the output format is jsonlines where each line has "paper_id" and "embedding" keys')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for prediction')
    parser.add_argument('--ctrl-token', default=None, help='Optional Control Token')
    parser.add_argument('--mode', default=None, help='Optional IR mode for Query Candidates data samples')

    args = parser.parse_args()
    model_dir = args.model_dir
    if args.ctrl_token:
        print(f"Using control token: {args.ctrl_token}")
    if args.mode == "ir":
        dataset = QueryDataset(data_path=args.data_path, model_dir=model_dir, batch_size=args.batch_size,
                               ctrl_token=args.ctrl_token)
    else:
        dataset = Dataset(data_path=args.data_path, model_dir=model_dir, batch_size=args.batch_size,
                          ctrl_token=args.ctrl_token)
    model = Model(0 if not dataset.ctrl_token else 1, model_dir)
    results = {}
    try:
        for batch, batch_ids in tqdm(dataset.batches(), total=len(dataset) // args.batch_size):
            emb = model(batch)
            for paper_id, embedding in zip(batch_ids, emb.unbind()):
                results[paper_id] = {"paper_id": paper_id, "embedding": embedding.detach().cpu().numpy().tolist()}
            del batch
            del emb
            # torch.cuda.empty_cache()
    except Exception as e:
        print(e)
    finally:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as fout:
            for res in results.values():
                fout.write(json.dumps(res) + '\n')
        print("Results size:" + str(len(results)))


if __name__ == '__main__':
    main()