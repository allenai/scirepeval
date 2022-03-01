from pl_training import PhantasmLight
import glob
from subprocess import Popen
import os
import time

task = "mesh_search"
parent_dir = f"/net/nfs2.s2-research/phantasm/phantasm_new/lightning_logs/co_train/s2and_search/checkpoints/"
# chkpt = f'{parent_dir}/ep-epoch=1_avg_val_loss-avg_val_loss=0.447-v1.ckpt'

chkpts = glob.glob(parent_dir + "/*.ckpt")
print(chkpts)
# for i, chkpt in enumerate(chkpts):
# model = PhantasmLight.load_from_checkpoint(chkpt, batch_size=16, use_ctrl_tokens=False, )
# model.encoder.save_pretrained(parent_dir+f'hf{i}/model/')
# model.tokenizer.save_pretrained(parent_dir+f'hf{i}/tokenizer/')
# model.tokenizer.save_vocabulary(parent_dir+f'hf{i}/tokenizer/')
os.system(f'../specter/scripts/embed.sh {parent_dir} s2and_search')