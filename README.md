
<div align="center">
 <img src="https://github.com/DATEXIS/medBERT.de/assets/37253540/6dfec04d-73a7-403c-9098-c5032719b8e9" width="500">
 <p></p>
</div>

This is the repository for the paper - **[MEDBERT.de: A Comprehensive German BERT Model for the Medical Domain](https://arxiv.org/abs/2303.08179)**


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

### Using Docker
We also provide a Docker image with prebuilt dependencies here:
[medbert.de](https://hub.docker.com/r/pgrundmann/medbert.de)

This image contains all necessary dependencies and scripts to run the pre-training and evaluation of our medbert.de model.

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

## Availability
Our model is availble on [Huggingface](https://huggingface.co/GerMedBERT).
We provide two versions of the model, one pre-trained on deduplicated radiology data, the other one on the full dataset. 

## Pretraining

### Data

| Name            |   n_documents |   n_sentences |    n_words |   size_mb |
|:----------------|--------------:|--------------:|-----------:|----------:|
| DocCheck        |         63840 |        720404 |   12299257 |     91.95 |
| GGPONC          |          4369 |         66256 |    1194345 |      9.21 |
| Webcrawl        |         11322 |        635806 |    9323774 |     64.57 |
| Pubmed          |         12139 |        108936 |    1983752 |     15.96 |    
| Radiology       |       3657801 |      60839123 |  520717615 |   4195.07 |
| Spinger OA      |        257999 |      14183396 |  259284884 |   1985.57 |
| EHR             |        373421 |       4603461 |   69639020 |    439.85 |
| Doctoral theses |       7486    |       4665850 |   90380880 |    647.46 |
| Thieme          |        330994 |      10445580 |  186200935 |   2898.16 |
| Wiki            |          3639 |        161714 |    2799787 |     21.52 |
| Summary         |       4723010 |      96544947 | 1155945499 |  10386.02 |
 

### 🪄 Hyperparameters

#### Phase 1 *Short Sequences*

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
- [GottBERT](https://huggingface.co/uklfr/gottbert-base)
- [Multilingual BERT cased](https://huggingface.co/bert-base-multilingual-cased)

## Data
Due to data protection laws and privacy we can not open source most of the evaluation datasets. However in most of the cases you can request access to the data for research purposes.

Please contact the following people to request access to the corresponding datasets:

- GGPONC - [Florian Borchert](https://hpi.de/lippert/senior-researcher-labs/research-group-in-memory-computing-for-digital-health/florian-borchert.html)
- GraSCCO - [Florian Borchert](https://hpi.de/lippert/senior-researcher-labs/research-group-in-memory-computing-for-digital-health/florian-borchert.html)
- Radiology Benchmarks (WristNER, ChestCT Classification, Chest X-Ray Classification) - [Keno Bressem](https://radiologie.charite.de/metas/person/person/address_detail/pd_dr_med_keno_bressem/)
- OPS and ICD code classification tasks - [Moritz Augustin](https://tiplu.de/moritz-augustin-gf-tiplu/)


