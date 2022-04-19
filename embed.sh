#!/bin/bash

MODEL_DIR=$1
OUTPUT_SUFFIX=$2

HOME_DIR="/net/nfs2.s2-research"
EMBED_DIR="${HOME_DIR}/phantasm/phantasm_new"


# echo "Biomimicry"
# python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/biomimicry/test_new.jsonl --output ${HOME_DIR}/phantasm/biomimicry/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --ctrl-token '{"val": "[CLF]"}' --encoder-type "pals"


# echo "Search"
# python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/search/test_new_reqd.jsonl --output ${HOME_DIR}/phantasm/search/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --fields "title" "abstract" "venue" "year" --ctrl-token '{"val": {"query": "[QRY]", "candidates": "[SAL]"}}' --encoder-type "pals"


# echo "FoS"
# python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/fos/fos_test_new.json --output ${HOME_DIR}/phantasm/fos/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --ctrl-token '{"val": "[CLF]"}' --encoder-type "pals"



echo "Scidocs"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/specter2/paper_metadata_mag_mesh.json --output ${HOME_DIR}/phantasm/specter2/emb_phantasm_mag_mesh_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --ctrl-token '{"val": "[CLF]"}'  #--encoder-type "pals"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/specter2/paper_metadata_recomm.json --output ${HOME_DIR}/phantasm/specter2/emb_phantasm_recomm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --ctrl-token '{"val": "[SAL]"}' # --encoder-type "pals"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/specter2/paper_metadata_view_cite_read.json --output ${HOME_DIR}/phantasm/specter2/emb_phantasm_view_cite_read_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --ctrl-token '{"val": "[SAL]"}'  #--encoder-type "pals"


echo "Feeds"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/feeds/feeds_paper_query.jsonl --output ${HOME_DIR}/phantasm/feeds/emb_phantasm_paper_query_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --ctrl-token '{"val": "[SAL]"}'  #--encoder-type "pals"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/feeds/feeds_multipaper_query.jsonl --output ${HOME_DIR}/phantasm/feeds/emb_phantasm_multi_paper_query_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --ctrl-token '{"val": "[SAL]"}'  #--encoder-type "pals"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/feeds/feeds_name_query.jsonl --output ${HOME_DIR}/phantasm/feeds/emb_phantasm_name_query_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --ctrl-token '{"val": {"query": "[QRY]", "candidates": "[SAL]"}}'  #--encoder-type "pals"


echo "Citation Context Count"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/cite_context/test_reqd.json --output ${HOME_DIR}/phantasm/cite_context/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --ctrl-token '{"val": "[SAL]"}'  #--encoder-type "pals"


echo "Trec COVID"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/trec_covid/test.json --output ${HOME_DIR}/phantasm/trec_covid/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --ctrl-token '{"val": {"query": "[QRY]", "candidates": "[SAL]"}}'  #--encoder-type "pals"


echo "s2and"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/s2and/full_test_partitioned/test.jsonl --output ${HOME_DIR}/phantasm/s2and/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --batch-size 4 --ctrl-token '{"val": "[ATH]"}'  #--encoder-type "pals"


echo "MeSH"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/mesh_plus/test.json --output ${HOME_DIR}/phantasm/mesh/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --ctrl-token '{"val": "[CLF]"}'  #--encoder-type "pals"



