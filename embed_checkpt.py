from pl_training import PhantasmLight
import glob
from subprocess import Popen
import os
import time

# task = "mesh_search"
parent_dir = f"/net/nfs2.s2-research/phantasm/phantasm_new/lightning_logs/full_run/ro_after_specter/checkpoints/"
# chkpt = f'{parent_dir}/ep-epoch=3_avg_val_loss-avg_val_loss=0.299.ckpt'

chkpts = glob.glob(parent_dir + "/*.ckpt")
print(chkpts)
# for i, chkpt in enumerate(chkpts):
# model = PhantasmLight.load_from_checkpoint(chkpt, batch_size=16, use_ctrl_tokens=True, )
# model.encoder.save_pretrained(parent_dir+f'model/')
# model.tokenizer.save_pretrained(parent_dir+f'tokenizer/')
# model.tokenizer.save_vocabulary(parent_dir+f'tokenizer/')
os.system(f'./embed.sh {parent_dir} ro_after_specter')

# pd = "/net/nfs2.s2-research/phantasm/S2AND/s2and_mini/"
# for src in ["inspire", "kisti", "pubmed", "qian", "zbmath"]:
#     print(src)
#     cd = pd+src
#     os.system(f'python /net/nfs2.s2-research/phantasm/specter/scripts/embed_papers_hf.py --data-path {cd}/{src}_papers.json --output {cd}/src_phantasm.json')


