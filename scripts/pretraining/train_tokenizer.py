import datasets
from transformers import BertTokenizerFast

HOME = "/home/bressekk/Documents/medbert/"
if __name__ == "__main__":
    datasets = datasets.load_from_disk(HOME + "datasets/tokenizer_ready_dataset")


    def get_training_corpus():
        dataset = datasets["train"]
        BATCH_SIZE = 100_000
        for start_idx in range(0, len(dataset), BATCH_SIZE):
            samples = dataset[start_idx: start_idx + BATCH_SIZE]["text"]
            # samples = [line for line in samples["text"] if line is not None and len(line) > 0 and not line.isspace()]
            yield samples


    old_tokenizer = BertTokenizerFast.from_pretrained("bert-base-german-cased", use_fast=True)
    print("Initialized old tokenizer for training")

    tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), 30000)
    print("Trained new tokenizer")

    tokenizer.save_pretrained(HOME + "custom-german-tokenizer")
    print("Saved new tokenizer")
