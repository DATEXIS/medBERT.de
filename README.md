![medbert-banner](https://user-images.githubusercontent.com/37253540/225719392-cc72bdba-06d4-4436-83a2-22602bb7ce4f.png)


## Install requirements
### From YAML file
We provide the `medbert.yml` file to build a working conda environment to install most libraries.
To create an anaconda environment for medbert run:   

```bash
conda env create -f medbert.yml
```

Thils will create a virtual environment named `medbert` for you. 
However, medbert also requires nvidia-apex, and we found that it works better to install apex from source. 

```bash
conda activate medbert
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
Now you're all set to start working with medbert.

### From scratch
You can also install all libraries from scratch using conda, but this might lead to version conflicts. 

```bash
conda create -n medbert python=3.9
conda activate medbert
conda update -n base -c defaults conda -y
conda install -c pytorch -c nvidia pytorch-cuda=11.7 pytorch cudatoolkit=11.7 -y
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install transformers datasets tokenizers pytorch-lightning pandas tqdm jsonargparse[signatures]
pip install -U "jsonargparse[signatures]"
```

## Pretraining Corpora

| Name            |   n_documents |   n_sentences |    n_words |   size_mb |
|:----------------|--------------:|--------------:|-----------:|----------:|
| DocCheck        |         63840 |        720404 |   12299257 |     91.95 |
| GGPONC          |          4369 |         66256 |    1194345 |      9.21 |
| Webcrawl        |         11322 |        635806 |    9323774 |     64.57 |
| Pubmed          |         12139 |        108936 |    1983752 |     15.96 |
| Radiology       |       3657801 |      60839123 |  520717615 |   4195.07 |
| Spinger OA      |        257999 |      14183396 |  259284884 |   1985.57 |
| EHR             |        373421 |       4603461 |   69639020 |    439.85 |
| Doctoral theses |       3422391 |       4665850 |   90380880 |    647.46 |
| Thieme          |        330994 |      10445580 |  186200935 |   2898.16 |
| Wiki            |          3639 |        161714 |    2799787 |     21.52 |
| WMT22           |         11673 |        114421 |    2121250 |     16.7  |
| Summary         |       8149588 |      96544947 | 1155945499 |  10386.02 |
 

## 🪄 Pre-training Hyperparameters

### Phase 1 *Short Sequences*

| **Hyperparameter**              | **Value** |
|---------------------------------|-----------|
| Training Steps                  | 7038      |
| Learning Rate                   | 6e-3      |
| Sequence Length                 | 128       |
| Warmup Steps                    | 2000      |
| Optimizer                       | LAMB      |
| Precision                       | 16bit     |
| Batch Size                      | 8192      |
| Per GPU Batchsize (8*A100 80GB) | 256       | 
| Gradient Accumulation Steps     | 32        |
| #GPUs                           | 8         |

### Phase 2 *Long Sequences*

| **Hyperparameter**              | **Value** |
|---------------------------------|-----------|
| Training Steps                  | 1563      |
| Learning Rate                   | 4e-3      |
| Sequence Length                 | 512       |
| Warmup Steps                    | 200       |
| Optimizer                       | LAMB      |
| Precision                       | 16bit     |
| Batch Size                      | 4096      |
| Per GPU Batchsize (8*A100 80GB) | 32        | 
| Gradient Accumulation Steps     | 128       |
| #GPUs                           | 8         |


## Evaluation

**Evaluation is performed with the following models**
- GerMedBERT (ours)
- [gBERT](https://huggingface.co/bert-base-german-cased)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
- [German medBERT](https://huggingface.co/smanjil/German-MedBERT)
- [GerNerMedBERT](https://huggingface.co/jfrei/de_GERNERMEDpp_GermanBERT) (für NER Task only)
- [GottBERT](https://huggingface.co/uklfr/gottbert-base)

**TODO**

* Add results



### 👓 Knowledge Benchmark

### 1. MC-Questions Charité

### 🇩🇪 Common Language Benchmarks

#### 2. GermEval14
| **Model**              | **Task** | **F1** |
|----------------------|-----------|-----------|
| gBert                 | Validation    |87.38|
| Gottbert                 | Validation |-|
| (ours)MedBert                 | Validation  |74.84|

#### 3. GermEval18
| **Model**              | **Task** | **F1** |
|----------------------|-----------|-----------|
| gBert                 | Validation    |0.5257535576820374|
| Gottbert                 | Validation |-|
| MedBert                 | Validation  |-|
### 🏥 Clinical / Biomedical Benchmarks

#### 4. BRONCO150 and BRONCO50 (hidden test-set)

#### 5. N2C2 German
| **Model**              | **Task** | **Token F1** | **Token Recall** | **Token Precision** | **Token AUC** |
|------------------------|----------|--------------|------------------|---------------------|---------------|

|----------------------|-----------|-----------|-----------|-----------|-----------|
| gBert                 | Validation    |0.8349246382713318| - | - | - |
| Gottbert                 | Validation |0.860611081123352|- | - | - |
| MedBert                 | Validation  |0.8398441672325134|- | - | - |
| gBert                 | Test    |0.8304128646850586| 0.8213436603546143| 0.8441508412361145 | 0.959336519241333|
| Gottbert                 | Test |0.8529325127601624|0.8428363800048828| 0.8674018383026123 | 0.952163815498352|
| MedBert                 | Test  |**Token F1** |**Token Recall** | **Token Precision** | **Token AUC** |


#### 6. Classification ChestCT
| **Model**              | **Task** | **AUC** |
|----------------------|-----------|-----------|
| German BERT                 | Chest-CT      |93.05|
| MedBERT                 | Chest-CT      |94.26|


#### 7. 🩻 Chest-X-Ray
| **Model**              | **Task** | **AUC** |
|----------------------|-----------|-----------|
| German BERT                 | Chest-XRay     |83.34|
| MedBERT                 | Chest-XRay      |84.89|

#### 8. Radiology NER Wrist CT
| **Model**              | **Task** | **F1** |
|----------------------|-----------|-----------|
| gBert                 | Validation    |0.5102534890174866|
| Gottbert                 | Validation |0.5304228067398071|
| MedBert                 | Validation  | 0.5069772601127625|


#### 9. GGPonc

#### 10. WikiDiseases

#### 11. LifeLine Corpus German
| **Model**              | **Task** | **F1** |
|----------------------|-----------|-----------|
| gBert                 | Validation    |0.6418119668960571|
| Gottbert                 | Validation |0.6163371801376343|
| MedBert                 | Validation  | 0.5928230285644531|
