import json
import uuid
from tqdm import tqdm
import argparse

stopwords = "myself, we, our, ours, ourselves, you, your, yours, yourself, yourselves, he, him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, not, only, own, same, so, than, too, very, can, will, just, don, should, now".split(
    ", ")
stopwords = set(stopwords)

html = "background-color:, margin:".split(", ")
html = set(html)

def get_chunks(s, size=100):
    s_split = s.split()
    for i in range(0, len(s_split), size):
        yield " ".join(s_split[i: i + size])


def clean(s):
    return s.strip().strip('\u200e').replace('\u00ad', '').replace("\u200b", "").replace('\0', '').strip()

def load_data(filename, n=None):
    data = []
    with open(filename, "r") as fin:
        # lines = fin.readlines()
        for line in tqdm(fin, total=n):
            data.append(json.loads(line))
    return data

def store_data(filename, data):
    with open(filename, "w+") as outfile:
        for idx, element in enumerate(data):
            # print(round(idx * 100 / len(data), 2), "%", end="\r")
            # sys.stdout.flush()
            json.dump(element, outfile)
            outfile.write("\n")

def main(arg):

    in_data = load_data(args.input)
    out_data = []

    for inst in in_data:

        new_output = []

        for i, output in enumerate(inst["output"]):

            new_provenance_list = []

            for j, provenance in enumerate(output["provenance"]):

                if "chunk_id" in provenance and "text" not in provenance:
                    print("ERROR")
                    raise Exception("!")

                # print(provenance.keys())
                text = " ".join(provenance["text"].split())
                try:
                    title = " ".join(provenance["title"].split())
                except:
                    title = " ".join(provenance["url_title"].split())
                count_stopw = len(set(text.lower().split()).intersection(stopwords))
                count_html = any([h in text.lower() for h in html])

                # should this instance be filtered?
                if count_stopw < 5 or \
                        count_html or \
                        len(title.split()) > 50:
                    print("FILTERED")
                    #input("...")
                    continue

                
                for chunk in get_chunks(clean(provenance["text"])):
                    new_provenance_list.append({
                        "chunk_id": str(uuid.uuid4()),
                        "url": provenance["url"],
                        "url_title": clean(provenance["url_title"]),
                        "text": chunk
                    })

            if len(new_provenance_list) > 0:
                new_output.append({"provenance":new_provenance_list})
        
        if len(new_output) > 0:
            inst["output"] = new_output
            out_data.append(inst)

    store_data(args.output, out_data)
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="input file",
    )


    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="output file",
    )

    args = parser.parse_args()

    print(args, flush=True)

    main(args)