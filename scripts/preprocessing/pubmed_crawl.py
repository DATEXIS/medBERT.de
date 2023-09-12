import re
from urllib.request import urlopen
from metapub import PubMedFetcher
from tqdm import tqdm
import time


def abstract_from_pmid(pmid: int) -> str: 
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
    site = urlopen(url).read().decode().encode().decode("utf-8")
    ger_part = [x for x in site.split("lang") if x.startswith('="de"')]
    if len(ger_part) > 0: 
        abstract = re.search('(?<=content\=").+?(?=")', ger_part[0])
        return abstract
    
with open("german_abstracts.csv", "w+") as f: 
    f.write("pmid;text\n")
    
query = '("German"[Language]) AND (fha[Filter])'

retmax = 1000
restart = 0

while True: 
    fetch = PubMedFetcher()
    pmids = fetch.pmids_for_query(query, since=1960, retmax=retmax, retstart=restart)
    
    if pmids == []:
        break  # end of articles

    restart += retmax

    with open("german_abstracts.csv", "a") as f:
        for pmid in tqdm(pmids, postfix=f"articles {restart-retmax}-{restart}"):
            abstract = abstract_from_pmid(pmid)
            if abstract:
                f.write(f'{pmid};"{abstract.group(0)}"\n')

    with open("restart.txt", "w+") as f:
        f.write(str(restart))
    return restart


if __name__ == "__main__":

    query = '("German"[Language]) AND (fha[Filter])'
    retmax = 1000
    if os.path.exists("restart.txt"):
        with open("restart.txt", "r") as f:
            restart = int(f.read())
            print(restart)
    else:
        restart = 39000

    if not os.path.exists("german_abstracts.csv"):
        with open("german_abstracts.csv", "w+") as f:
            f.write("pmid;text\n")

    while True:
        try:
            restart = main(query, restart, retmax)
        except Exception as e:
            print(f"Exception raised. Going to sleep for 10 seconds {e}")
            for _ in tqdm(range(10)):
                time.sleep(1)
        if restart == "End of articles":
            print("============================== Done ==============================")
            break

    abstracts = pd.read_csv("german_abstracts.csv", sep=";")
    abstracts["text"] = [remove_html_tags(string) for string in tqdm(abstracts.text)]
    abstracts.to_csv("german_abstracts_clean.csv", index=False)
