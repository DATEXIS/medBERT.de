import ast
import xml.etree.ElementTree as ET

import pandas as pd


def get_doi(front):
    article_id = front.findall(".//article-id")
    book_id = front.findall(".//book-id")
    book_part_id = front.findall(".//book-part-id")
    if article_id:
        doi = article_id[0].text

    elif book_id:
        doi = book_id[0].text

    elif book_part_id:
        doi = book_part_id[0].text

    return doi


def get_abstract(front):
    abstracts = front.findall(".//abstract/p")
    abstract = ""
    # part = ""
    for abstract_y in abstracts:

        abstract = abstract + " ".join(abstract_y.itertext()) + " "

    return abstract


def clean_string(s1):
    # replaces new line, unbreakable space, TAB with SPACE and strip the final string
    # also replaces multiple spaces with a single one
    if s1 is None:
        return None
    s2 = s1.replace("\n", " ").replace("\u00a0", " ").replace("\t", " ").strip()
    return " ".join(s2.split())


def xml_attributes(root):
    articles = root[4]

    for idx in range(len(articles)):
        article = articles[idx]
        for i, front in enumerate(article):

            doi = get_doi(front)
            # print(doi)
            abstract = clean_string(get_abstract(front))
            yield doi, abstract


if __name__ == "__main__":

    xml_base_path = "nbs/springer_xml_{}.txt"

    topics = [
        "BIOMEDICINE",
        "PHARMACY",
        "DENTISTRY",
        "LIFE_SCIENCES",
        "MEDICINE_PUBLIC_HEALTH",
        "MEDICINE_PUBLIC_HEALTH_2",
    ]

    xml_paths = [xml_base_path.format(topic) for topic in topics]

    for idx, xml_path in enumerate(xml_paths):
        with open(xml_path) as f:
            lines = f.readlines()

        counter = 0
        abstract_count = 0

        for ii, line in enumerate(lines):
            line_dict = ast.literal_eval(line)
            line_xml = line_dict["page"]
            root = ET.fromstring(line_xml)
            counter = counter + len(root[4])

            if idx == 0 and ii == 0:
                line_records = pd.DataFrame(xml_attributes(root)).rename(columns={0: "doi", 1: "abstract"})
                line_records = line_records[line_records.abstract != ""]
                line_records["primary_subject"] = topics[idx]
            else:
                curr = pd.DataFrame(xml_attributes(root)).rename(columns={0: "doi", 1: "abstract"})
                curr = curr[curr.abstract != ""]
                curr["primary_subject"] = topics[idx]
                line_records = pd.concat([line_records, curr])

        print(topics[idx], line_records.shape)
    line_records[~pd.isnull(line_records.abstract)].to_csv("springer_doi_abstracts.csv", index=False)
