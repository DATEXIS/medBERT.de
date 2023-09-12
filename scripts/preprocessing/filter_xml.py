import ast
import xml.etree.ElementTree as ET


# need to update notebook script currently only 100  records per request are possible
# check json facets key maybe easier
def get_article_meta(article_meta_atts, gather, article_id):
    gather[article_id] = dict()

    for meta_att in article_meta_atts:

        if meta_att[0].text == "journal-subject-primary":
            gather[article_id]["subject-primary"] = meta_att[1].text

        if meta_att[0].text == "journal-subject-secondary":
            if "subject-secondary" not in gather[article_id]:
                gather[article_id]["subject-secondary"] = list()
            gather[article_id]["subject-secondary"].append(meta_att[1].text)

    return gather


def get_book_meta(book_meta_atts, gather, book_id):
    gather[book_id] = dict()

    for meta_att in book_meta_atts:

        if meta_att[0].text == "book-subject-primary":
            gather[book_id]["subject-primary"] = meta_att[1].text

        if meta_att[0].text == "book-subject-secondary":
            if "subject-secondary" not in gather[book_id]:
                gather[book_id]["subject-secondary"] = list()
            gather[book_id]["subject-secondary"].append(meta_att[1].text)

    return gather


def xml_attributes(root, gather, lost, ii):
    articles = root[4]

    for idx in range(len(articles)):
        article = articles[idx]
        for i, front in enumerate(article):
            # print(len(front))
            if len(front) == 2:
                # print('==2')
                article_meta = front[1]
                # article_id = article_meta.findall('article-id')[0].text
                meta_atts = article_meta.findall("custom-meta-group")
                # journal_meta = front[0]

                if len(meta_atts):
                    article_id = article_meta.findall("article-id")[0].text
                    meta_atts = meta_atts[0]
                    get_article_meta(meta_atts, gather, article_id)
                    yield gather, article_id

                else:
                    book_part_meta = front[0]
                    # title_group = book_part_meta[0]
                    # label = title_group[0].text
                    # title = title_group[1].text

                    book_body = front[1]
                    book_part = book_body[0]
                    book_part_meta = book_part[0]

                    # book_part_id = book_part_meta.findall('book-part-id')[0].text
                    # part_title = book_part_meta[1][0].text

                    self_uri = book_part_meta.findall("self-uri")
                    links = list()

                    for uri in self_uri:
                        links.append(uri.text)

            elif len(front) > 2:
                # print('>2')
                book_id = front.findall("book-id")[0].text
                meta_atts = front.findall("custom-meta-group")
                meta_atts = meta_atts[0]
                get_book_meta(meta_atts, gather, book_id)
                yield gather, book_id
            else:
                # print('ELSE')
                ### hier kommen artikel vor aus den Buechern die im Abschnitt davor abgefangen werden.
                #### es gibt keine primary oder secondary subjects hier
                book_part_meta = front[0]
                # book_part_id = book_part_meta.findall("book-part-id")[0].text
                # cmeta = book_part_meta.findall("custom-meta-group")[0]

                # lost[f'page_{ii}'] = front
            yield gather, -1


if __name__ == "__main__":

    xml_path = "nbs/springer_xml_MEDICINE_PUBLIC_HEALTH.txt"
    secondary_filter_path = "secondary_filter.txt"
    gather = dict()
    lost = dict()

    primaries = set()
    secondaries = set()

    with open(xml_path) as f:
        lines = f.readlines()

    with open(secondary_filter_path) as f:
        sec_filter = f.readlines()
        sec_filter = list(map(lambda x: x[:-1], sec_filter))

    counter = 0

    for ii, line in enumerate(lines):
        line_dict = ast.literal_eval(line)
        line_xml = line_dict["page"]
        root = ET.fromstring(line_xml)
        counter = counter + len(root[4])

        with open("spinger_DOIS_medicine_public_health.txt", "a+") as f:
            for gather, doi in xml_attributes(root, gather, lost, ii):
                if doi != -1:
                    if doi == "10.1007/s00391-021-02007-1":
                        pass
                    primary_subject = gather[doi]["subject-primary"]
                    secondary_subject = set(gather[doi]["subject-secondary"])
                    # print(primary_subject, secondary_subject)
                    primaries.add(primary_subject)
                    secondaries = secondaries.union(secondary_subject)
                    # if secondary_subject.intersection(sec_filter):
                    #    print(secondary_subject)
                    # if doi == '10.1007/978-3-642-73442-7_2':
                    #    print(doi)
                    # f.write(doi + '\n')

    print(len(gather), counter)

    with open("springer_subject_primary.txt", "w") as f:
        for p in primaries:
            f.write(p + "\n")

    with open("springer_subject_secondary.txt", "w") as f:
        for p in secondaries:
            f.write(p + "\n")
