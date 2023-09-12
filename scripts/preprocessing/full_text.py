import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timedelta

import pandas as pd
from tqdm import tqdm

# https://spdi.public.springernature.app/xmldata/jats?q=%20(doi:10.1007/s11825-012-0314-3%20OR%20doi:10.1007/s11825-012-0316-1)&excludeElements=ref-list&api_key=4ad34d397a4a3402f467259e48ba9c2c/sndeal-api&s=1&p=3  # noqa E501
# url = f"https://spdi.public.springernature.app/xmldata/jats?q=%20(doi:{row.doi}%20OR%20doi:{})&excludeElements=ref-list,contrib-group&api_key=4ad34d397a4a3402f467259e48ba9c2c/sndeal-api"  # noqa E501

# take all dois and split in groups of 100 dois. query!


def chunks(lst, n=50):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield list(lst.values[i : i + n])


def api_call(doi_df):

    for idx, row in doi_df.iterrows():
        url = (
            f"https://spdi.public.springernature.app/xmldata/jats?q=doi:{row.doi}"
            + "&excludeElements=ref-list,contrib-group&api_key=4ad34d397a4a3402f467259e48ba9c2c/sndeal-api"
        )
        response = urllib.request.urlopen(url)
        page = response.read().decode("utf8")
        yield page, row


def bulk_api_call(doi_df):

    start_url = "https://spdi.public.springernature.app/xmldata/jats?q=%20("
    add_doi_url = "%20OR%20" + "doi:{}"
    end_url = ")&excludeElements=ref-list&api_key=4ad34d397a4a3402f467259e48ba9c2c/sndeal-api&s=1&p=50"
    dois = list(chunks(doi_df.doi))
    for doi_chunk in tqdm(dois):
        tmp_url = start_url + f"doi:{doi_chunk[0]}"
        for doi_idx in range(1, len(doi_chunk)):
            tmp_url = tmp_url + add_doi_url.format(doi_chunk[doi_idx])
        tmp_url = tmp_url + end_url
        response = urllib.request.urlopen(tmp_url)
        page = response.read().decode("utf8")
        yield page, doi_df.primary_subject.iloc[0]


def process_response(page, row):

    root = ET.fromstring(page)
    result = root[3]
    records = root[4]

    total = result.findall(".//total")[0].text

    if int(total):
        # records = records.findall('.//body')
        if records:
            # text = " ".join(records[0].itertext())
            dict_page = dict()
            dict_page["page"] = page
            page = json.dumps(dict_page)
            with open(f"springer_xml_full_text_{row.primary_subject}.txt", "a+") as f:
                f.write(page + "\n")

    else:
        with open("lost_full_text_dois.txt", "a+") as f:
            f.write(",".join([str(row.name), row.doi, row.primary_subject]) + "\n")


if __name__ == "__main__":

    # path = 'spinger_DOIS_medicine_public_health.txt'

    doi_path = "springer_doi.csv"
    doi_df = pd.read_csv(doi_path)
    doi_df["doi"] = doi_df["doi"].str.replace("\n", "")
    start_doi = "10.1007/BF01722243"
    start = doi_df[doi_df.doi == start_doi].index.item()

    print("start from index", start)
    primary_subject = "BIOMEDICINE"

    doi_df = doi_df[doi_df.primary_subject == primary_subject]
    doi_df = doi_df.loc[min((start + 1), len(doi_df)) :].drop_duplicates("doi", keep="first")

    start = time.time()
    api_calls = 0

    # for page, row in tqdm(api_call(doi_df)):
    for page, primary_subject in tqdm(bulk_api_call(doi_df)):
        elapsed = timedelta(seconds=time.time() - start).total_seconds()
        api_calls = api_calls + 1

        if api_calls == 5000:
            print("reached daily limit")
            print("index", api_calls - 1)
            break

        if api_calls % 149 == 0 and api_calls > 0:
            if (60 - elapsed) > 0:
                time.sleep(60 - elapsed)
            start = time.time()

        dict_page = dict()
        dict_page["page"] = page
        page = json.dumps(dict_page)

        with open(f"springer_xml_Afull_text_{primary_subject}.txt", "a+") as f:
            f.write(page + "\n")
