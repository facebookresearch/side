import copy
import csv
import json
import os.path
import uuid
from pathlib import Path

import hydra
import tqdm
from omegaconf import OmegaConf

from dpr.dataset.utils import resolve_file

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


def create_mixed_index_data(
        cfg,
        TRAIN_DPR_RETRIEVED,
        TRAIN_BM25_RETRIEVED,
        TRAIN_GOLD_SPLIT_DATA_FILE,
        VALID_DPR_RETRIEVED,
        VALID_BM25_RETRIEVED,
        VALID_GOLD_SPLIT_DATA_FILE,
        OUT_CSV_FILE,
):
    """
    Create index data / Mixing ccnet passages and generated train passages

    1. Retrieve passages with DPR and BM25 from big index for the training split and the validation split

    Only ~7000 URLs are in big index -> create positive passages from training data

    2. Create index data from those passages

    :param DPR_RETRIEVED_TRAIN_SPLIT_DATA_FILE:
    :param BM25_RETRIEVED_TRAIN_SPLIT_DATA_FILE:
    :param GOLD_TRAIN_SPLIT_DATA_FILE:
    :return:
    """
    out = list()

    def get_jlines_dict(FILE):
        jlines = dict()
        with open(FILE) as f:
            for i, (line) in enumerate(tqdm.tqdm(f)):
                jline = json.loads(line)
                jlines[jline["id"]] = jline
        return jlines

    for DPR_RETRIEVED, BM25_RETRIEVED, GOLD_SPLIT_DATA_FILE in [
        [TRAIN_DPR_RETRIEVED, TRAIN_BM25_RETRIEVED, TRAIN_GOLD_SPLIT_DATA_FILE],
        [VALID_DPR_RETRIEVED, VALID_BM25_RETRIEVED, VALID_GOLD_SPLIT_DATA_FILE],
    ]:

        jlines_dpr = get_jlines_dict(DPR_RETRIEVED)
        jlines_bm25 = get_jlines_dict(BM25_RETRIEVED)
        jlines_gold = get_jlines_dict(GOLD_SPLIT_DATA_FILE)

        all_keys = set(jlines_dpr.keys())
        all_keys = all_keys.intersection(set(jlines_bm25.keys()))
        all_keys = all_keys.intersection(set(jlines_gold.keys()))

        print("len(all_keys)", len(all_keys))

        for id in tqdm.tqdm(all_keys):

            jline_dpr = jlines_dpr[id]
            jline_bm25 = jlines_bm25[id]
            jline_gold = jlines_gold[id]

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
                    if count_stopw < 5 or \
                            count_html or \
                            len(title.split()) > 50:
                        do_filter = True
                        break

            if do_filter:
                # print(json.dumps(jline_gold, indent=4))
                continue

            positive_ctxs = list()
            negative_ctxs = list()

            true_urls = set()

            for output in jline_gold["output"]:
                if "provenance" not in output:
                    continue
                if len(output["provenance"]) == 0:
                    continue
                if len(output["provenance"])>0 and "text" not in output["provenance"][0]:
                    continue

                for provenance in output["provenance"]:
                    text, url = provenance["text"], provenance["url"],

                    if "wikipedia_title" in provenance:
                        title = clean(provenance["wikipedia_title"])
                    else:
                        title = clean(provenance["title"])

                    true_urls.add(provenance["url"])

                    for chunk in get_chunks(clean(provenance["text"])):
                        positive_ctxs.append({
                            "id": str(uuid.uuid4()),
                            "doc_id": 0,
                            "passage_id": 0,
                            "url": provenance["url"],
                            "title": title,
                            "text": chunk,
                            "system_hit": False,
                        })

            for jline_sys, is_bm25 in [(jline_dpr, False), (jline_bm25, True)]:
                for output in jline_sys["output"]:
                    if "provenance" not in output:
                        continue
                    if len(output["provenance"]) == 0:
                        continue
                    if "chunk_id" not in output["provenance"][0]:
                        continue

                    for provenance in output["provenance"]:
                        chunk_id, text, url = provenance["chunk_id"], provenance["text"], provenance["url"],

                        if "wikipedia_title" in provenance:
                            title = provenance["wikipedia_title"]
                        else:
                            title = provenance["title"]

                        if url in true_urls:
                            continue

                        negative_ctxs.append(
                            {
                                "id": chunk_id,
                                "doc_id": 0,
                                "passage_id": 0,
                                "url": url,
                                "title": clean(title),
                                "text": clean(text),
                                "system_hit": is_bm25,
                            }
                        )
                        # chunk_ids.add(chunk_id)

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
                # " ".join(section.split()).split("::::")[1],
                " ".join(before_cit.split()[-cfg.create_data.word_citation_prefix:] + ["[CIT]"] + after_cit.split()[:cfg.create_data.word_citation_suffix])
            ])

            train_instance["positive_ctxs"] = positive_ctxs
            train_instance["negative_ctxs"] = negative_ctxs

            if len(positive_ctxs) > 0:
                out.append(train_instance)

    with open(OUT_CSV_FILE, "w") as f_out:

        chunk_id_memory = set()

        csv_writer = csv.writer(f_out, delimiter="\t")

        for inst in tqdm.tqdm(out):
            for ctxts in ['positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs', 'hard_negative_doc_ctxs']:
                if ctxts in inst:
                    for ctxt in inst[ctxts]:
                        if ctxt["id"] not in chunk_id_memory:
                            chunk_id_memory.add(ctxt["id"])
                            csv_writer.writerow(
                                [
                                    ctxt["id"],
                                    clean(ctxt["text"]),
                                    clean(ctxt["title"]),
                                    ctxt["url"],
                                    ctxt["passage_id"],
                                    ctxt["system_hit"],
                                ])


@hydra.main(config_path="../../conf", config_name="create_data/reproduce_reranker_training_index")
def main(cfg):

    if cfg.create_data.output_directory is None or len(cfg.create_data.output_directory) == 0:
        print(f"'output_directory' was not set. Please run this script with {__file__} output_directory=PATH_TO_OUTPUT_DIR")
        exit()

    GOLD_VALID_SPLIT_DATA_FILE = resolve_file(cfg, cfg.create_data.gold_valid_file)
    RETRIEVED_BM25_VALID_SPLIT_DATA_FILE = resolve_file(cfg, cfg.create_data.retrieved_bm25_valid_file)
    RETRIEVED_DPR_VALID_SPLIT_DATA_FILE = resolve_file(cfg, cfg.create_data.retrieved_dpr_valid_file)

    for gold_train_file, retrieved_bm25_train_file, retrieved_dpr_train_file in zip(
            cfg.create_data.gold_train_file,
            cfg.create_data.retrieved_bm25_train_file,
            cfg.create_data.retrieved_dpr_train_file
    ):

        GOLD_TRAIN_SPLIT_DATA_FILE = resolve_file(cfg, gold_train_file)
        RETRIEVED_BM25_TRAIN_SPLIT_DATA_FILE = resolve_file(cfg, retrieved_bm25_train_file)
        RETRIEVED_DPR_TRAIN_SPLIT_DATA_FILE = resolve_file(cfg, retrieved_dpr_train_file)

        os.makedirs(cfg.create_data.output_directory, exist_ok=True)

        key = f"{Path(cfg.create_data.output_directory).stem.replace('-', '_')}_{Path(GOLD_TRAIN_SPLIT_DATA_FILE).stem.replace('-', '_')}"
        datasets_retrieval_file = str(Path(hydra.utils.to_absolute_path(__file__)).parent.parent.parent) + "/conf/datasets/retrieval.yaml"
        datasets_retrieval = OmegaConf.load(datasets_retrieval_file)

        if key in datasets_retrieval and os.path.exists(os.path.join(cfg.create_data.output_directory, Path(GOLD_TRAIN_SPLIT_DATA_FILE).stem + ".csv")):
            print(f"Dataset with key {key} already exists. Nothing more to do.")
        else:
            if not os.path.exists(os.path.join(cfg.create_data.output_directory, Path(GOLD_TRAIN_SPLIT_DATA_FILE).stem + ".csv")):
                create_mixed_index_data(
                    cfg,
                    GOLD_TRAIN_SPLIT_DATA_FILE,
                    RETRIEVED_BM25_TRAIN_SPLIT_DATA_FILE,
                    RETRIEVED_DPR_TRAIN_SPLIT_DATA_FILE,
                    GOLD_VALID_SPLIT_DATA_FILE,
                    RETRIEVED_BM25_VALID_SPLIT_DATA_FILE,
                    RETRIEVED_DPR_VALID_SPLIT_DATA_FILE,
                    OUT_CSV_FILE=os.path.join(cfg.create_data.output_directory, Path(GOLD_TRAIN_SPLIT_DATA_FILE).stem + ".csv"),
                )
        datasets_retrieval[key] = {
            "file": os.path.join(cfg.create_data.output_directory, Path(GOLD_TRAIN_SPLIT_DATA_FILE).stem + ".csv"),
            "_target_": "dpr.dataset.retrieval.WaferCsvCtxSrc",
        }
        with open(datasets_retrieval_file, "w") as f:
            OmegaConf.save(datasets_retrieval, f)

if __name__ == "__main__":
    main()
