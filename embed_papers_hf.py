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
from bert_pals import BertPalsEncoder
from adapter_fusion import AdapterEncoder, AdapterFusion


class Dataset:

    def __init__(self, data_path, model_dir, max_length=512, batch_size=32, ctrl_token=None, fields=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir + "/tokenizer/")  # model_dir + "/tokenizer/"
        self.max_length = max_length
        self.batch_size = batch_size
        self.ctrl_token = ctrl_token
        if not fields:
            fields = ["title", "abstract"]
        self.fields = fields
        # data is assumed to be a json file
        # [{"corpus_id":...,"title":..., "abstract":...}...]
        try:
            self.data = json.load(open(data_path, "r"))
        except:
            with open(data_path) as f:
                self.data = [json.loads(line) for line in f]
        if type(self.data) == dict:
            self.data = list(self.data.values())
        print("Special tokens:")
        print(self.tokenizer.all_special_tokens)

    def __len__(self):
        return len(self.data)

    def batches(self):
        # create batches
        batch = []
        batch_ids = []
        batch_size = self.batch_size
        seen_ids = set()
        i = 0
        count = 0
        if "corpus_id" in self.data[0]:
            key = "corpus_id"
        else:
            key = "paper_id"
        for d in self.data:
            if key in d:
                bid = d[key]
                if bid not in seen_ids:
                    seen_ids.add(bid)
                    text = "" if not self.ctrl_token else self.ctrl_token + " "
                    if not d['title']:
                        print(d)
                    text += d['title'] + ' '
                    if d.get('abstract'):
                        text += f'{self.tokenizer.sep_token} ' + d.get('abstract')

                    if (i) % batch_size != 0 or i == 0:
                        batch_ids.append(d[key])
                        batch.append(text)
                    else:
                        input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                                   return_tensors="pt", max_length=self.max_length)
                        yield input_ids.to('cuda'), batch_ids
                        batch_ids = [d[key]]
                        batch = [text]
                    i += 1
            else:
                count += 1
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
        seen_ids = set()
        i = 0
        for d in self.data:
            d_pair = [d["query"]] + d["candidates"]
            for j, dp in enumerate(d_pair):
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
                if key and key not in seen_ids:
                    seen_ids.add(key)
                    if self.ctrl_token:
                        if type(self.ctrl_token) == dict:
                            if j == 0:
                                ctrl_token = self.ctrl_token["query"]
                            else:
                                ctrl_token = self.ctrl_token["candidates"]
                        else:
                            ctrl_token = self.ctrl_token
                        text = ctrl_token + " " + text
                    qc = "q" if j == 0 else "c"
                    if ((i) % batch_size != 0) or i == 0:
                        batch_ids.append((key, qc))
                        batch.append(text)
                    else:
                        input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                                   return_tensors="pt", max_length=self.max_length)
                        yield input_ids.to('cuda'), batch_ids
                        batch_ids = [(key, qc)]
                        batch = [text]
                    i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt", max_length=self.max_length)
            input_ids = input_ids.to('cuda')
            yield input_ids, batch_ids


class Model:

    def __init__(self, encoder, token_idx, model_dir, task_id=None):
        self.model = encoder
        self.model.to('cuda')
        self.model.eval()
        self.token_idx = token_idx
        self.task_id = task_id
        print(self.task_id)

    def __call__(self, input_ids, batch_ids):
        if not self.task_id or type(self.task_id) != dict:
            output = self.model(**input_ids) if not self.task_id else self.model(task_id=self.task_id,
                                                                                 x=input_ids["input_ids"])
        else:
            x = input_ids["input_ids"]
            output = torch.zeros(x.shape[0], x.shape[1], 768).to("cuda")
            q_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "q"])
            c_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "c"])

            if not q_idx.shape[0]:
                output = self.model(task_id=self.task_id["candidates"], x=input_ids["input_ids"])
            else:
                for i, v in enumerate(sorted(self.task_id.values())):
                    curr_input_idx = q_idx if v == "[QRY]" else c_idx
                    curr_input = x[curr_input_idx]
                    curr_output = self.model(task_id=v, x=curr_input)
                    try:
                        output[curr_input_idx] = curr_output
                    except:
                        output[curr_input_idx] = curr_output.last_hidden_state
        try:
            return output.last_hidden_state[:, self.token_idx, :]  # cls token
        except:
            return output[:, self.token_idx, :]  # cls token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='path to a json file containing paper metadata')
    parser.add_argument('--model-dir', help='path to the HF model and tokenizer')
    parser.add_argument('--model-name', help='Base HF model for adapters', default=None)
    parser.add_argument('--output', help='path to write the output embeddings file. '
                                         'the output format is jsonlines where each line has "paper_id" and "embedding" keys')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for prediction')
    parser.add_argument('--ctrl-token', default=None, help='Optional Control Token', type=json.loads)
    parser.add_argument('--mode', default=None, help='Optional IR mode for Query Candidates data samples')
    parser.add_argument('--fields', nargs='+', help='Fields other than title, abstract', default=None)
    parser.add_argument('--encoder-type', help='Encoder type from default/pals/adapters', default="default")
    parser.add_argument('--format', help='Format json or pkl', default="json")

    args = parser.parse_args()
    model_dir = args.model_dir
    model_name = args.model_name
    ctrl_token = None
    task_id = None
    if args.ctrl_token:
        if args.encoder_type != "default":
            task_id = args.ctrl_token['val']
        else:
            ctrl_token = args.ctrl_token['val']
            print(f"Using control token: {ctrl_token}")
    if args.mode == "ir":
        dataset = QueryDataset(data_path=args.data_path, model_dir=model_dir, batch_size=args.batch_size,
                               ctrl_token=ctrl_token, fields=args.fields)
    else:
        dataset = Dataset(data_path=args.data_path, model_dir=model_dir, batch_size=args.batch_size,
                          ctrl_token=ctrl_token)
    if args.encoder_type == "default":
        encoder = AutoModel.from_pretrained(model_dir + "/model/")
        model = Model(encoder, 0 if not dataset.ctrl_token else 1, model_dir)
    elif args.encoder_type == "pals":
        task_ids = []
        if args.ctrl_token:
            task_ids = ["[ATH]", "[CLF]", "[QRY]", "[SAL]"]
        encoder = BertPalsEncoder(config=f"{model_dir}/model/config.json", task_ids=task_ids,
                                  checkpoint=f"{model_dir}/model/pytorch_model.bin")
        model = Model(encoder, 0 if not ctrl_token else 1, model_dir, task_id=task_id)
    elif args.encoder_type == "adapters":
        t_ids = [task_id] if type(task_id) != dict else list(task_id.values())
        encoder = AdapterEncoder(model_name, t_ids, load_dir=f"{model_dir}/model/adapters/")
        model = Model(encoder, 0 if not ctrl_token else 1, model_dir, task_id=task_id)
    elif args.encoder_type == "fusion":
        t_ids = ["[ATH]", "[CLF]", "[QRY]", "[SAL]"] if args.ctrl_token else None
        encoder = AdapterFusion(model_name, t_ids, adapters_dir=f"{model_dir}/model", inference=True)
        model = Model(encoder, 0 if not ctrl_token else 1, model_dir, task_id=task_id)
    results = {}
    try:
        for batch, batch_ids in tqdm(dataset.batches(), total=len(dataset) // args.batch_size):
            emb = model(batch, batch_ids)
            for paper_id, embedding in zip(batch_ids, emb.unbind()):
                if type(paper_id) == tuple:
                    paper_id = paper_id[0]
                results[paper_id] = {"paper_id": paper_id, "embedding": embedding.detach().cpu().numpy().tolist()}
            del batch
            del emb
            # torch.cuda.empty_cache()
    except Exception as e:
        print(e)
    finally:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if args.format == "json":
            with open(args.output, 'w') as fout:
                for res in results.values():
                    fout.write(json.dumps(res) + '\n')
        else:
            import numpy as np
            import pickle
            paper_ids, embs = np.array([str(v["paper_id"]) for v in results.values()]), np.array(
                [v["embedding"] for v in results.values()])
            pickle.dump((embs, paper_ids), open(args.output, "wb"))
        print("Results size:" + str(len(results)))


if __name__ == '__main__':
    main()
