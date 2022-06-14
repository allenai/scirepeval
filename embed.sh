#!/bin/bash

MODEL_DIR=$1
OUTPUT_SUFFIX=$2
ENCODER_TYPE=$3
MODEL_NAME=$4

HOME_DIR="/net/nfs2.s2-research"
EMBED_DIR="${HOME_DIR}/phantasm/phantasm_new"



# echo "Biomimicry"
# python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/biomimicry/test_new.jsonl --output ${HOME_DIR}/phantasm/biomimicry/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[CLF]"}' --model-name ${MODEL_NAME}


echo "DRSM"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/drsm/test.json --output ${HOME_DIR}/phantasm/drsm/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[CLF]"}' --model-name ${MODEL_NAME}


echo "FoS"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/fos/fos_test_new.json --output ${HOME_DIR}/phantasm/fos/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[CLF]"}' --model-name ${MODEL_NAME}



echo "Scidocs"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/specter2/paper_metadata_mag_mesh.json --output ${HOME_DIR}/phantasm/specter2/emb_phantasm_mag_mesh_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[CLF]"}'  --model-name ${MODEL_NAME}
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/specter2/paper_metadata_recomm.json --output ${HOME_DIR}/phantasm/specter2/emb_phantasm_recomm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[SAL]"}' --model-name ${MODEL_NAME}
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/specter2/paper_metadata_view_cite_read.json --output ${HOME_DIR}/phantasm/specter2/emb_phantasm_view_cite_read_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[SAL]"}' --model-name ${MODEL_NAME}


echo "Feeds"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/feeds/feeds_paper_query.jsonl --output ${HOME_DIR}/phantasm/feeds/emb_phantasm_paper_query_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[SAL]"}' --model-name ${MODEL_NAME}
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/feeds/feeds_multipaper_query.jsonl --output ${HOME_DIR}/phantasm/feeds/emb_phantasm_multi_paper_query_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --batch-size 4 --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[SAL]"}' --model-name ${MODEL_NAME}
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/feeds/feeds_name_query.jsonl --output ${HOME_DIR}/phantasm/feeds/emb_phantasm_name_query_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": {"query": "[QRY]", "candidates": "[SAL]"}}' --model-name ${MODEL_NAME}


echo "Search"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/search/test_new_reqd.jsonl --output ${HOME_DIR}/phantasm/search/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --fields "title" "abstract" "venue" "year" --batch-size 4 --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": {"query": "[QRY]", "candidates": "[SAL]"}}' --model-name ${MODEL_NAME}


echo "Citation Context Count"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/cite_context/test_reqd.json --output ${HOME_DIR}/phantasm/cite_context/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --batch-size 4 --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[SAL]"}' --model-name ${MODEL_NAME}


echo "Trec COVID"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/trec_covid/test.json --output ${HOME_DIR}/phantasm/trec_covid/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": {"query": "[QRY]", "candidates": "[SAL]"}}' --model-name ${MODEL_NAME}


echo "Peer review"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/peer_review/nips_test_complete.json --output ${HOME_DIR}/phantasm/peer_review/emb_phantasm_nips_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[ATH]"}' --model-name ${MODEL_NAME}

python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/peer_review/icip_test_complete.json --output ${HOME_DIR}/phantasm/peer_review/emb_phantasm_icip_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[ATH]"}' --model-name ${MODEL_NAME}

echo "s2and"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/s2and/full_test_partitioned/test.jsonl --output ${HOME_DIR}/phantasm/s2and/emb_phantasm_${OUTPUT_SUFFIX}.json --mode ir --model-dir ${MODEL_DIR} --batch-size 4 --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[ATH]"}' --model-name ${MODEL_NAME}


echo "MeSH"
python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/scidocs/data/mesh_plus/test.json --output ${HOME_DIR}/phantasm/mesh/emb_phantasm_${OUTPUT_SUFFIX}.json --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --ctrl-token '{"val": "[CLF]"}' --model-name ${MODEL_NAME}


echo "s2and_mini"
blocks=("arnetminer" "inspire" "kisti" "qian" "pubmed" "zbmath")
for block in ${blocks[@]}; do
    echo ${block}
    python ${EMBED_DIR}/embed_papers_hf.py --data-path ${HOME_DIR}/phantasm/S2AND/s2and_mini/${block}/${block}_papers.json  --output ${HOME_DIR}/phantasm/S2AND/s2and_mini/${block}/${block}_${OUTPUT_SUFFIX}.pkl --model-dir ${MODEL_DIR} --encoder-type ${ENCODER_TYPE} --format "pkl" --ctrl-token '{"val": "[ATH]"}'
done



