from evaluation.embeddings_generator import EmbeddingsGenerator
from evaluation.eval_datasets import SimpleDataset, IRDataset

tasks = {"TREC-CoVID": {"name":"TREC-CoVID","type":"adhoc_search","data":{"meta":{"name":"allenai/scirepeval","config":"trec_covid"},"test":{"name":"allenai/scirepeval_test","config":"trec_covid"}},"metrics":["ndcg"]}}
model = "Qwen/Qwen3-4B"
dataset = IRDataset(('allenai/scirepeval', 'trec_covid'), 1)
save_path = "/mount/weka/shriya/embeddings/qwen_4b_generic/TREC-CoVID"
if os.path.exists(save_path):
    results = EmbeddingsGenerator.load_embeddings_from_jsonl(save_path)
else:
    results = dict()

try:
    for batch, batch_ids in tqdm(dataset.batches(), total=len(dataset) // dataset.batch_size):
        for paper_id in batch_ids:
            if type(paper_id) == tuple:
                paper_id = paper_id[0]
            if paper_id not in results:
                with torch.no_grad():
                    emb = model(batch, batch_ids)
                results[paper_id] = emb.detach().cpu().numpy()
                del emb
        del batch
        torch.cuda.empty_cache()
except Exception as e:
    logger.error("Exception in generating embeddings", exc_info=e)
finally:
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as fout:
        for k, v in results.items():
            fout.write(json.dumps({"doc_id": k, "embedding": v.tolist()}) + '\n')
logger.info(f"Generated {len(results)} embeddings")