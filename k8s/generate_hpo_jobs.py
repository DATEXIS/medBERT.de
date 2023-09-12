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
TASK_DIRS = ["./germeval-18", "./germeval-14", "./chest-ct", "./chest-xray", "./wrist_ner", "n2c2_ner"]


def main():
    for task_dir in TASK_DIRS:
        for model in MODELNAMES:
            generated_directory = task_dir + "-generated/" + model.replace("/", "-")
            if not os.path.exists(generated_directory):
                os.makedirs(generated_directory)

            for filename in glob(task_dir + "/*.yaml"):
                with open(filename, "r") as f:
                    job_ = f.read()
                job_ = job_.replace("{{modelname}}", model)
                job_ = job_.replace("{{jobname}}", model.replace("/", "-").lower())

                target_filename = filename.split("/")[-1]
                with open(generated_directory + "/" + target_filename, "w+") as target_file:
                    target_file.write(job_)


if __name__ == "__main__":
    main()
