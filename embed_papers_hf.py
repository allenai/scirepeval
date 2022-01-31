# simple script for embedding papers using huggingface Specter
# requirement: pip install --upgrade transformers==4.2.2
from transformers import AutoModel, AutoTokenizer
import json
import argparse
from tqdm.auto import tqdm
import pathlib
import torch


class Dataset:

    def __init__(self, data_path, max_length=512, batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/net/nfs2.s2-research/phantasm/phantasm_new/lightning_logs/full_run/mixed_prop/checkpoints/tokenizer/')
        self.max_length = max_length
        self.batch_size = batch_size
        # data is assumed to be a json file
        # [{"corpus_id":...,"title":..., "abstract":...}...]
        try:
            self.data = json.load(open(data_path, "r"))
        except:
            with open(data_path) as f:
                self.data = [json.loads(line) for line in f]

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
                if (i) % batch_size != 0 or i == 0:
                    batch_ids.append(d["corpus_id"])
                    batch.append(d['title'] + ' ' + (d.get('abstract') or ''))
                else:
                    input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                               return_tensors="pt", max_length=self.max_length)
                    yield input_ids.to('cuda'), batch_ids
                    batch_ids = [d["corpus_id"]]
                    batch = [d['title'] + ' ' + (d.get('abstract') or '')]
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
                if "corpus_id" in dp:
                    if (i) % batch_size != 0 or i == 0:
                        batch_ids.append(dp["corpus_id"])
                        batch.append(dp['title'] + ' ' + (dp.get('abstract') or ''))
                    else:
                        input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                                   return_tensors="pt", max_length=self.max_length)
                        yield input_ids.to('cuda'), batch_ids
                        batch_ids = [dp["corpus_id"]]
                        batch = [dp['title'] + ' ' + (dp.get('abstract') or '')]
                    i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt", max_length=self.max_length)
            input_ids = input_ids.to('cuda')
            yield input_ids, batch_ids


class Model:

    def __init__(self):
        self.model = AutoModel.from_pretrained(
            '/net/nfs2.s2-research/phantasm/phantasm_new/lightning_logs/full_run/mixed_prop/checkpoints/model/')
        self.model.to('cuda')
        self.model.eval()

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        return output.last_hidden_state[:, 0, :]  # cls token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='path to a json file containing paper metadata')
    parser.add_argument('--output', help='path to write the output embeddings file. '
                                         'the output format is jsonlines where each line has "paper_id" and "embedding" keys')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for prediction')

    args = parser.parse_args()
    dataset = QueryDataset(data_path=args.data_path, batch_size=args.batch_size)
    model = Model()
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
