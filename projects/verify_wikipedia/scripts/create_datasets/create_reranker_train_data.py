import copy
import csv
import json
import uuid
from collections import defaultdict

import hydra
import tqdm

stopwords = "myself, we, our, ours, ourselves, you, your, yours, yourself, yourselves, he, him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, not, only, own, same, so, than, too, very, can, will, just, don, should, now".split(
    ", ")
stopwords = set(stopwords)

html = "background-color:, margin:".split(", ")
html = set(html)


def should_filter(text: str, title: str):
    count_stopw = len(set(text.lower().split()).intersection(stopwords))
    count_html = any([h in text.lower() for h in html])
    # should this instance be filtered?
    if count_stopw < 5 or \
            count_html or \
            len(title.split()) > 50:
        return True
    return False


def get_chunks(s, size=100):
    s_split = s.split()
    for i in range(0, len(s_split), size):
        yield " ".join(s_split[i: i + size])


def clean(s):
    return s.strip().strip('\u200e').replace('\u00ad', '').replace("\u200b", "").replace('\0', '').strip()[:4000]


def interleave_rankings(rankings):
    result = list()
    max_len = max(len(rankings["dpr"]), len(rankings["bm25"]))
    for i in range(max_len):
        if i < len(rankings["dpr"]):
            result.append(rankings["dpr"][i])
        if i < len(rankings["bm25"]):
            result.append(rankings["bm25"][i])
    return result


def create_training_data(
        cfg,
        DPR_RETRIEVED,
        BM25_RETRIEVED,
        GOLD_SPLIT_DATA_FILE,
        OUT_DIR,
        out_file_name,
):

        out = list()

        with open(DPR_RETRIEVED) as f_dpr:
            with open(BM25_RETRIEVED) as f_bm25:
                with open(GOLD_SPLIT_DATA_FILE) as f_gold:

                    line_bm25 = next(f_bm25)

                    for i, (line_gold, line_dpr) in enumerate(tqdm.tqdm(zip(f_gold, f_dpr))):

                        jline_dpr = json.loads(line_dpr)
                        jline_bm25 = json.loads(line_bm25)
                        jline_gold = json.loads(line_gold)

                        try:
                            while jline_dpr["id"] != jline_bm25["id"]:
                                # print(jline_dpr["id"], jline_bm25["id"])
                                # print('jline_dpr["id"] != jline_bm25["id"]')
                                line_bm25 = next(f_bm25)
                                jline_bm25 = json.loads(line_bm25)
                            line_bm25 = next(f_bm25)
                        except:
                            continue

                        do_filter = False

                        for output in jline_gold["output"]:
                            if "provenance" not in output:
                                continue
                            for provenance in output["provenance"]:
                                if "text" not in provenance or "chunk_id" in provenance:
                                    continue

                                if "wikipedia_title" in provenance:
                                    title = provenance["wikipedia_title"]
                                else:
                                    title = provenance["title"]

                                text = " ".join(provenance["text"].split())
                                title = " ".join(title.split())
                                count_stopw = len(set(text.lower().split()).intersection(stopwords))
                                count_html = any([h in text.lower() for h in html])

                                # should this instance be filtered?
                                if count_stopw < 3 or \
                                        count_html or \
                                        len(title.split()) > 50:
                                    do_filter = True
                                    break

                        if do_filter:
                            # print(json.dumps(jline_gold, indent=4))
                            continue

                        true_urls = set()

                        for output in jline_gold["output"]:
                            if "provenance" not in output:
                                continue
                            if len(output["provenance"]) == 0 or "text" not in output["provenance"][0]:
                                continue
                            for provenance in output["provenance"]:
                                true_urls.add(provenance["url"])

                        positive_ctxs = defaultdict(list)
                        negative_ctxs = defaultdict(list)

                        for jline_sys, retriever_str in [(jline_dpr, "dpr"), (jline_bm25, "bm25")]:
                            for output in jline_sys["output"]:
                                if "provenance" not in output:
                                    continue
                                if len(output["provenance"]) == 0:
                                    continue
                                if "chunk_id" not in output["provenance"][0]:
                                    continue

                                for i, provenance in enumerate(output["provenance"][:cfg.create_data.top_k_retrieved_per_retriever]):
                                    chunk_id, url = provenance["chunk_id"], provenance["url"],

                                    if retriever_str == "dpr":
                                        text = provenance["text"]
                                    elif retriever_str == "bm25":
                                        try:
                                            text = json.loads(provenance["text"])["contents"]
                                        except:
                                            continue
                                    else:
                                        raise Exception(f"System unknown {retriever_str}")

                                    if "wikipedia_title" in provenance:
                                        title = provenance["wikipedia_title"]
                                    else:
                                        title = provenance["title"]

                                    if url in true_urls:
                                        positive_ctxs[retriever_str].append(
                                            {
                                                "id": chunk_id,
                                                "doc_id": 0,
                                                "passage_id": 0,
                                                "url": url,
                                                "title": clean(title),
                                                "text": clean(text),
                                                "system_hit": retriever_str,
                                                "rank": i,
                                            }
                                        )
                                    else:
                                        negative_ctxs[retriever_str].append(
                                            {
                                                "id": chunk_id,
                                                "doc_id": 0,
                                                "passage_id": 0,
                                                "url": url,
                                                "title": clean(title),
                                                "text": clean(text),
                                                "system_hit": retriever_str,
                                                "rank": i,
                                            }
                                        )

                        instance_template = {
                            "dataset": "wafer-kiltweb_dpr",
                            "question": "",
                            "answers": list(),
                            "positive_ctxs": list(),
                            "negative_ctxs": list(),
                            "negative_doc_ctxs": list(),
                            "hard_negative_ctxs": list(),
                            "hard_negative_doc_ctxs": list(),
                        }

                        train_instance = copy.deepcopy(instance_template)
                        SEP_split = jline_gold["input"].split("[SEP]")
                        if len(SEP_split) != 3:
                            print(jline_gold["input"])
                            continue
                        title, section, ctxt = SEP_split
                        before_cit, after_cit = ctxt.split("[CIT]")
                        train_instance["question"] = " [SEP] ".join([
                            " ".join(title.split()),
                            " ".join(section.split()),
                            " ".join(before_cit.split()[-cfg.create_data.word_citation_prefix:] + [" [CIT] "] + after_cit.split()[:cfg.create_data.word_citation_suffix])
                        ])

                        train_instance["positive_ctxs"] = interleave_rankings(positive_ctxs)
                        train_instance["negative_ctxs"] = interleave_rankings(negative_ctxs)

                        if len(train_instance["positive_ctxs"]) > 0:
                            out.append(train_instance)

        print(len(out))

        with open(
                f"{OUT_DIR}/{out_file_name}",
                "w") as f_out:
            for i in range(len(out)):
                f_out.writelines(json.dumps(out[i]) + "\n")

@hydra.main(config_path="../../conf", config_name="create_data/reproduce_reranker_training_data")
def main(cfg):
    DPR_RETRIEVED_TRAIN_SPLIT_DATA_FILE = cfg.create_data.retrieved_dpr_train_file
    BM25_RETRIEVED_TRAIN_SPLIT_DATA_FILE = cfg.create_data.retrieved_bm25_train_file
    GOLD_TRAIN_SPLIT_DATA_FILE = cfg.create_data.gold_train_file

    DPR_RETRIEVED_VALID_SPLIT_DATA_FILE = cfg.create_data.retrieved_dpr_train_file
    BM25_RETRIEVED_VALID_SPLIT_DATA_FILE = cfg.create_data.retrieved_bm25_train_file
    GOLD_VALID_SPLIT_DATA_FILE = cfg.create_data.gold_train_file

    for DPR_RETRIEVED, BM25_RETRIEVED, GOLD_SPLIT_DATA_FILE, out_file_name in [
        [DPR_RETRIEVED_TRAIN_SPLIT_DATA_FILE, BM25_RETRIEVED_TRAIN_SPLIT_DATA_FILE, GOLD_TRAIN_SPLIT_DATA_FILE, "wafer-train-kiltweb.jsonl"],
        [DPR_RETRIEVED_VALID_SPLIT_DATA_FILE, BM25_RETRIEVED_VALID_SPLIT_DATA_FILE, GOLD_VALID_SPLIT_DATA_FILE, "wafer-dev-kiltweb.jsonl"],
    ]:
        create_training_data(
            cfg,
            DPR_RETRIEVED,
            BM25_RETRIEVED,
            GOLD_SPLIT_DATA_FILE,
            OUT_DIR=cfg.create_data.output_directory,
            out_file_name=out_file_name,
        )

if __name__ == "__main__":
    main()
