import os
from glob import glob

MODELNAMES = [
    "uklfr/gottbert-base",
    "GerMedBERT/medbert-512",
    "smanjil/German-MedBERT",
    "GerMedBERT/halbgott-in-weiss-base",
    "bert-base-multilingual-cased",
    "SCAI-BIO/bio-gottbert-base",
    "GerMedBERT/medbert-512-no-duplicates"
]


def main():
    test_template_jobs = glob("./*.yaml")
    for task_dir in test_template_jobs:
        generated_directory = task_dir.split(".")[1][1:] + "-generated/"
        if not os.path.exists(generated_directory):
            os.makedirs(generated_directory)
        for model in MODELNAMES:
            with open(task_dir, "r") as f:
                job_ = f.read()
            job_ = job_.replace("{{modelname}}", model)
            job_ = job_.replace("{{jobname}}", model.replace("/", "-").lower())
            job_ = job_.replace("{{jobname}}", model.replace("_", "-").lower())

            target_filename = generated_directory + model.replace("/", "-") + ".yaml"
            with open(target_filename, "w+") as target_file:
                target_file.write(job_)


if __name__ == "__main__":
    main()
