
## Inference

This guide provides the steps to generate relevant document embeddings with SciRepEval models.
 
 ### Step 1 Create a Model instance
 ```python
from evaluation.encoders import Model

#Base/MTL CLS
model = Model(variant="default", base_checkpoint="allenai/specter")

#MTL CTRL
model = Model(variant="default", base_checkpoint="allenai/scirepeval_ctrl", use_ctrl_codes=True)

#PALs
model = Model(variant="pals", base_checkpoint="allenai/scirepeval_pals", all_tasks=["[CLF]", "[QRY]", "[RGN]", "[PRX]"])

#Adapters
adapters_dict = {"[CLF]": "allenai/scirepeval_adapters_clf", "[QRY]": "allenai/scirepeval_adapters_qry", "[RGN]": "allenai/scirepeval_adapters_rgn", "[PRX]": "allenai/scirepeval_prx"}
model = Model(variant="adapters", base_checkpoint="malteos/scincl", adapters_load_from=adapters_dict, all_tasks=["[CLF]", "[QRY]", "[RGN]", "[PRX]"])

#Fusion
model = Model(variant="fusion", base_checkpoint="malteos/scincl", adapters_load_from=adapters_dict, fusion_load_from=<fusion chkpoint directory>, all_tasks=["[CLF]", "[QRY]", "[RGN]", "[PRX]"])

```

### Step 2 Determine task type
Choose the relevant task id value from the below python dict keyed on task type
``TASK_IDS = {"classification": "[CLF]", "regression": "[RGN]", "proximity": "[PRX]",  
  "adhoc_search": {"query": "[QRY]", "candidates": "[PRX]"}}``

```python
model.task_id = "[CLF]" #OR "[RGN]"/"[PRX]"/{{"query": "[QRY]", "candidates": "[PRX]"}}}
```

For feeding raw text input to the model, follow step 3. If working with a specific dataset jump to Step 4.

### Step 3 Generate embeddings for raw text
Use the model object as a callable.
```python
embeddings = model("Attention is all you need[SEP]Attention is all you need")
```

### Step 4 Generate embeddings for a dataset

- If data instances consists of records with fields: eg. 
```json
{
    "corpus_id": 22715986,
    "title": "Accuracy of MRI for treatment response assessment after taxane- and anthracycline-based neoadjuvant chemotherapy in HER2-negative breast cancer.",
    "abstract": "BACKGROUND\nStudies suggest that MRI is an accurate means for assessing tumor size after neoadjuvant chemotherapy (NAC). However, accuracy might be dependent on the receptor status of tumors. MRI accuracy for response assessment after homogenous NAC in a relative large group of patients with stage II/III HER2-negative breast cancer has not been reported before.\n\n\nMETHODS\n250 patients from 26 hospitals received NAC (docetaxel, adriamycin and cyclophosphamide) in the context of the NEOZOTAC trial. MRI was done after 3 cycles and post-NAC. Imaging (RECIST 1.1) and pathological (Miller and Payne) responses were recorded."
}
```
```python
from evaluation.eval_datasets import SimpleDataset  
from evaluation.evaluator import Evaluator

dataset = ("allenai/scirepeval", "biomimicry") #OR path like "scirepeval/biomimicry/test.json"
evaluator = Evaluator(name="biomimcry", dataset, SimpleDataset, model, batch_size=32, fields=["title", "abstract"], key="paper_id")
embeddings = evaluator.generate_embeddings(save_path="embeddings.json")
```
- If data instances consists of query-candidate pairs: eg. 
```json
{
    "dataset": "aminer",
    "query":
    {
        "corpus_id": 24254880,
        "title": "[Characteristics of heavy metal elements and their relationship with magnetic properties of river sediment from urban area in Lanzhou].",
        "abstract": "The contents of As, Co, Cr, Cu, Ni, Pb, V and Zn in the surface sediments from 8 rivers in urban area in Lanzhou were monitored by ecological risk which was assessed by the potential ecological Håkanson index, and the index of geoaccumulation (Igeo), sediment enrichment factor (R), and environmental magnetism. The results showed that: (1) the potential ecological risk of heavy metals of As, Co, Ni, V in surface sediments from 8 rivers were low, which belonged to low ecological risk. But the risk of heave metals Cr, Pb, Zn in surface sediments from Yuer river was high, which belonged to middle ecological risk, and in downstream of Yuer river, the element of Cu belonged to high ecological risk. (2) The rivers in Lanzhou could be divided into four groups according to the heavy mental pollution degree: first type, such as Paihong river, Shier river, Yuer river and Shuimo river, called downstream concentrate type; second type, such as Qili river, called upstream concentrate type; third type, such as Luoguo river and Dasha river, called less affected type; fourth type, Lanni river, which polluted heavily in up and downstream; (3) The correlation analysis between magnetic parameters and element contents show that the parameters which mainly reflect the concentration of the magnetic minerals (X, SIRM, Ms) have close association with Cr, Ni, Pb, Zn, Cu, So we can infer that the magnetic minerals in deposits samples mainly came from electroplating effluent, motor vehicle emission, and domestic sewage. SIRM/X shows a strong correlation with Cr, Ni, Pb, Zn, indicating the distribution of anthropogenic particulates. (4) The magnetic minerals(X, SIRM, Ms) have a strong correlation with the geoaccumulation (Igeo) than potential ecological risk index and enrichment factor (R). These results suggest a possible approach for source identification of magnetic material in pollution studies and the validity of using magnetic measurements to mapping the polluted area."
    },
    "candidates":
    [
        {
            "corpus_id": 12540419,
            "title": "Combination of magnetic parameters and heavy metals to discriminate soil-contamination sources in Yinchuan--a typical oasis city of Northwestern China.",
            "abstract": "Various industrial processes and vehicular traffic result in harmful emissions containing both magnetic minerals and heavy metals. In this study, we investigated the levels of magnetic and heavy metal contamination of topsoils from Yinchuan city in northwestern China. The results demonstrate that magnetic mineral assemblages in the topsoil are dominated by pseudo-single domain (PSD) and multi-domain (MD) magnetite. The concentrations of anthropogenic heavy metals (Cr, Cu, Pb and Zn) and the magnetic properties of χlf, SIRM, χARM, and 'SOFT' and 'HARD' remanence are significantly correlated, suggesting that the magnetic minerals and heavy metals have common sources. Combined use of principal components and fuzzy cluster analysis of the magnetic and chemical data set indicates that the magnetic and geochemical properties of the particulates emitted from different sources vary significantly. Samples from university campus and residential areas are mainly affected by crustal material, with low concentrations of magnetic minerals and heavy metals, while industrial pollution sources are characterized by high concentrations of coarse magnetite and Cr, Cu, Pb and Zn. Traffic pollution is characterized by Pb and Zn, and magnetite. Magnetic measurements of soils are capable of differentiating sources of magnetic minerals and heavy metals from industrial processes, vehicle fleets and soil parent material.",
            "score": 1
        }
    ]...
}
```

```python
from evaluation.eval_datasets import IRDataset  
from evaluation.evaluator import Evaluator

dataset = ("allenai/scirepeval", "feeds_1") #OR path like "scirepeval/feeds_1/test.json"
evaluator = Evaluator(name="biomimcry", dataset, IRDataset, model, batch_size=32, fields=["title", "abstract"], key="doc_id")
embeddings = evaluator.generate_embeddings(save_path="embeddings.json")
```






