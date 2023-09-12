## Evaluation Scripts

This section contains information about the evaluation scripts and Kubernetes job definitions used to reproduce the results of hyperparameter optimization and final evaluation on the test datasets.

### Hyperparameter Optimization and Evaluation Scripts

- For classification tasks, you can use the script `scripts/evaluation/hpo_radiology_classification.py`.
- For Named Entity Recognition (NER) tasks, utilize `scripts/evaluation/hpo_ner_ct.py`.

Each of these scripts requires the respective datasets to be provided via command-line arguments for evaluation.

### Kubernetes Job Definitions

The Kubernetes job file definitions essential for running these evaluation scripts can be found in the `k8s` directory.

We have used templates and a generator script, `k8s/generate_hpo_jobs.py`, to automatically generate these job definitions. However, please note that you may need to manually adjust the paths to the datasets in the template base files as per your specific setup and directory structure.
