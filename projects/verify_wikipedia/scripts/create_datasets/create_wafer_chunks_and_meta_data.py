import json
import uuid

import spacy
import tqdm

nlp = spacy.load("en_core_web_sm")


def clean(s):
    return s.strip().strip('\u200e').replace('\u00ad', '').replace("\u200b", "").replace('\0', '').strip()


def get_chunks(s, size=100):
    s_split = s.split()
    for i in range(0, len(s_split), size):
        yield " ".join(s_split[i: i + size])


if __name__ == "__main__":

    IN_DATA_PATH = "/checkpoint/fabiopetroni/WAI/time_split_data/train_cit_ALL_20210701.jsonl"
    OUT_DATA_PATH = "/checkpoint/fabiopetroni/WAI/time_split_data/train_cit_ALL_20210701_chunked+sents.jsonl"
    NR_PREV_SENTS = 4

    out = list()

    with open(IN_DATA_PATH) as f:
        for line in tqdm.tqdm_notebook(f):
            jline = json.loads(line.strip())
            title, section, text = jline["input"].split("[SEP]")
            prev_cit_text = text.split("[CIT]")[0]
            doc = nlp(prev_cit_text)
            jline["meta"]["sentences"] = [str(s) for s in list(doc.sents)[-NR_PREV_SENTS:]]

            provenance = list()
            for chunk in get_chunks(jline["meta"]["url_text"]):
                provenance.append({
                    "chunk_id": str(uuid.uuid4()),
                    "url": jline["output"][0]["original_url"],
                    "title": jline["output"][0]["title"],
                    "text": chunk,
                })

            new_jline = {
                "id": jline["id"],
                "input": jline["input"],
                "output": [
                    {
                        "answer": jline["output"][0]["original_url"],
                        "provencance": provenance
                    }
                ],
            }
            out.append(json.dumps(new_jline) + "\n")

    with open(OUT_DATA_PATH, "w") as f:
        f.writelines(out)

