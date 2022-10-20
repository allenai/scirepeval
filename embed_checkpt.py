import glob
import os

# task = "mesh_search"
version = ["specter_fusion"]
for v in version:
    print(v)
    parent_dir = f"/net/nfs2.s2-research/phantasm/phantasm_new/lightning_logs/full_run/{v}/checkpoints/"
    # chkpt = f'{parent_dir}/ep-epoch=3_avg_val_loss-avg_val_loss=0.276.ckpt'

    chkpts = glob.glob(parent_dir + "/*.ckpt")
    print(chkpts)
    # for i, chkpt in enumerate(chkpts):
    # model = PhantasmLight.load_from_checkpoint(chkpt, batch_size=16, use_ctrl_tokens=True, )
    #     model = PhantasmLight.load_from_checkpoint(chkpt, batch_size=16, lr=5e-5,
    #                               tokenizer="malteos/scincl",
    #                               model="malteos/scincl",
    #                               use_ctrl_tokens=False, pals_cfg=None, warmup_steps=400, adapter_type="single", log_dir=parent_dir)

    #     model.encoder.save_pretrained(parent_dir+'model/', adapter_names=["[ATH]", "[QRY]", "[CLF]"])
    # model.tokenizer.save_pretrained(parent_dir+f'tokenizer/')
    # model.tokenizer.save_vocabulary(parent_dir+f'tokenizer/')
    # torch.save(model.encoder.state_dict(),
    #                        parent_dir +'model/pytorch_model.bin')
    # model.encoder.bert_config.save_pretrained(parent_dir +'model/')
    os.system(f'./embed.sh {parent_dir} {v} fusion allenai/specter')

# pd = "/net/nfs2.s2-research/phantasm/S2AND/s2and_mini/"
# for src in ["inspire", "kisti", "pubmed", "qian", "zbmath"]:
#     print(src)
#     cd = pd+src
#     os.system(f'python /net/nfs2.s2-research/phantasm/specter/scripts/embed_papers_hf.py --data-path {cd}/{src}_papers.json --output {cd}/src_phantasm.json')