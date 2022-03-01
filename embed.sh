#!/bin/bash

MODEL_DIR=$1
OUTPUT_SUFFIX=$2

HOME_DIR="/net/nfs2.s2-research"
EMBED_DIR="${HOME_DIR}/phantasm/specter/scripts"

# python ${EMBED_DIR}/embed_papers_hf_new.py --data-path ${HOME_DIR}/phantasm/biomimicry/test_new.jsonl --output ${HOME_DIR}/phantasm/biomimicry/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR}

# python ${EMBED_DIR}/embed_papers_hf_new.py --data-path ${HOME_DIR}/phantasm/search/test_new_reqd.jsonl --output ${HOME_DIR}/phantasm/search/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR}

#python ${EMBED_DIR}/embed_papers_hf_new.py --data-path ${HOME_DIR}/scidocs/data/mesh_plus/test.json --output ${HOME_DIR}/phantasm/mesh/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR}

#python ${EMBED_DIR}/embed_papers_hf_new.py --data-path ${HOME_DIR}/phantasm/fos/fos_test.json --output ${HOME_DIR}/phantasm/fos/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR}

python ${EMBED_DIR}/embed_papers_hf_new.py --data-path ${HOME_DIR}/scidocs/data/s2and/test.jsonl --output ${HOME_DIR}/phantasm/s2and/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR}

# python ${EMBED_DIR}/embed_papers_hf_new.py --data-path ${HOME_DIR}/phantasm/specter2/paper_metadata_mag_mesh.json --output ${HOME_DIR}/phantasm/specter2/emb_phantasm_mag_mesh_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR}
# python ${EMBED_DIR}/embed_papers_hf_new.py --data-path ${HOME_DIR}/phantasm/specter2/paper_metadata_recomm.json --output ${HOME_DIR}/phantasm/specter2/emb_phantasm_recomm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR}
# python ${EMBED_DIR}/embed_papers_hf_new.py --data-path ${HOME_DIR}/phantasm/specter2/paper_metadata_view_cite_read.json --output ${HOME_DIR}/phantasm/specter2/emb_phantasm_view_cite_read_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR}

