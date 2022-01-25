import csv
import os
import pickle
import glob
import json
import uuid
import copy

import hydra
import numpy
import tqdm
from collections import Counter, defaultdict

from dpr.dataset.utils import resolve_file

stopwords = "myself, we, our, ours, ourselves, you, your, yours, yourself, yourselves, he, him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, not, only, own, same, so, than, too, very, can, will, just, don, should, now".split(", ")
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

def clean(s):
    return s.strip().strip('\u200e').replace('\u00ad', '').replace("\u200b", "").replace('\0', '').strip()

def get_chunks(s, size=100):
    s_split = s.split()
    for i in range(0, len(s_split), size):
        yield " ".join(s_split[i: i + size])


def get_instance_id__to__hard_negative_urls__map(BM25_RETRIEVED_DATA, cfg):
    count_urls = list()
    instance_id__to__hard_negative_urls__map = defaultdict(list)
    chunk_id__to__instance_id__map = defaultdict(list)
    with open(BM25_RETRIEVED_DATA) as f:
        for line in tqdm.tqdm(f):
            jline = json.loads(line)
            for i, output in enumerate(jline["output"]):
                for j, provenance in enumerate(output["provenance"][:30]):
                    if "text" in provenance:
                        assert len(provenance[cfg.create_data.id_field].split("-")) == 5, f"provenance['title'] did not contain a UUID in the form xxxxxxx-xxxx-xxxx-xxxx-xxxx this was a bug in the data; len was {len(provenance['title'].split('-'))}"
                        if provenance["url"] not in instance_id__to__hard_negative_urls__map[jline["id"]]:
                            instance_id__to__hard_negative_urls__map[jline["id"]].append(provenance["url"])
                        if jline["id"] not in chunk_id__to__instance_id__map[provenance[cfg.create_data.id_field]]:
                            chunk_id__to__instance_id__map[provenance[cfg.create_data.id_field]].append(jline["id"])  # title contains chunk id (bug in data)
            count_urls.extend(instance_id__to__hard_negative_urls__map[jline["id"]])

    count_urls_dict = Counter(count_urls)
    return instance_id__to__hard_negative_urls__map, count_urls_dict, chunk_id__to__instance_id__map


def get_url__to__chunks__map__from_hard_negatives(
        chunk_id__to__url__map,
        ignore_urls=None,
        count_urls_dict=None,
        chunk_id__to__instance_id__map=None,
        CHUNKS_GLOB=None,
):
    """
    :param chunk_id__to__url__map:
    :param ignore_urls:
    :param count_urls_dict:
    :param CHUNKS_GLOB:
    :return:
    """

    url__to__chunks__map = defaultdict(list)
    all_urls = set()
    chunk_files = list(glob.glob(CHUNKS_GLOB))
    for chunk_file in chunk_files:
        with open(chunk_file) as f:
            for i, line in enumerate(tqdm.tqdm(f)):
                chunk_id, text, title, chunk_hash, some_id = line.strip().split("\t")
                url = chunk_id__to__url__map[chunk_id]
                if url not in ignore_urls and url in count_urls_dict:
                    all_urls.add(url)
                    url__to__chunks__map[url].append(
                        {
                            "id": chunk_id,
                            "doc_id": len(all_urls) - 1,
                            "url": url,
                            "title": clean(title),
                            "text": clean(text),
                            "bm25_hit": chunk_id__to__instance_id__map[
                                chunk_id] if chunk_id in chunk_id__to__instance_id__map else None,
                            "passage_id": len(url__to__chunks__map[url])
                        }
                    )

    return url__to__chunks__map, all_urls


def get_url__to__passages_from_split_data(
        split_data,
        MAX_INSTANCES_FOR_TRAINING = -1,
):
    """
    Iterate through the training data and:
     - filter out instances with no urls, non-english url content,
     - collect a mapping from urls to passages

    :param split_data:
    :return:
    """

    meta_data = {
        "filtered": 0,
        "ids": set()
    }

    url__to__passages_from_split_data = dict()

    with open(split_data) as f_in:
        for inst, line in enumerate(tqdm.tqdm(f_in)):

            if MAX_INSTANCES_FOR_TRAINING > 0 and len(meta_data["ids"]) == MAX_INSTANCES_FOR_TRAINING:
                break

            jline = json.loads(line.strip())
            for i, output in enumerate(jline["output"]):

                if "chunk_id" in output["provenance"][0]:
                    # there is already a list of chunks in provenance

                    pass_rows_list = list()

                    for j, provenance in enumerate(output["provenance"]):

                        text = " ".join(provenance["text"].split())
                        title = " ".join(provenance["title"].split())
                        count_stopw = len(set(text.lower().split()).intersection(stopwords))
                        count_html = any([h in text.lower() for h in html])

                        # should this instance be filtered?
                        if count_stopw < 5 or \
                                count_html or \
                                len(title.split()) > 50:
                            meta_data["filtered"] += 1
                            continue
                        else:
                            pass_rows_list.append({
                                "id": provenance["chunk_id"],
                                "doc_id": str(len(url__to__passages_from_split_data)),
                                "url": provenance["url"],
                                "title": clean(provenance["title"]),
                                "text": clean(provenance["text"]),
                                "bm25_hit": False,
                                "passage_id": len(pass_rows_list),
                            })

                    if len(pass_rows_list) > 0:
                        meta_data["ids"].add(jline["id"])
                        url__to__passages_from_split_data[provenance["url"]] = pass_rows_list

                elif "text" in output["provenance"][0]:
                    # there is only a text in provenance and we have to create the chunks

                    for j, provenance in enumerate(output["provenance"]):

                        text = " ".join(provenance["text"].split())
                        title = " ".join(provenance["title"].split())
                        count_stopw = len(set(text.lower().split()).intersection(stopwords))
                        count_html = any([h in text.lower() for h in html])

                        # should this instance be filtered?
                        if count_stopw < 5 or \
                                count_html or \
                                len(title.split()) > 50:
                            meta_data["filtered"] += 1
                            continue

                        pass_rows_list = list()
                        for chunk in get_chunks(clean(provenance["text"])):
                            pass_rows_list.append({
                                "id": str(uuid.uuid4()),
                                "doc_id": str(len(url__to__passages_from_split_data)),
                                "url": provenance["url"],
                                "title": clean(provenance["title"]),
                                "text": chunk,
                                "bm25_hit": False,
                                "passage_id": len(pass_rows_list)
                            })

                        if len(pass_rows_list) > 0:
                            meta_data["ids"].add(jline["id"])
                            url__to__passages_from_split_data[provenance["url"]] = pass_rows_list

    return url__to__passages_from_split_data, meta_data


def _get_url_chunks_negatives(url_chunks, jline_id, nr_negatives, chunk_bm25_hit_index=None):
    result = list()
    for i, chunk in enumerate(copy.deepcopy(url_chunks)):
        if isinstance(chunk["bm25_hit"], set) or isinstance(chunk["bm25_hit"], list):
            chunk["bm25_hit"] = jline_id in chunk["bm25_hit"]
        else:
            chunk["bm25_hit"] = False
        if chunk_bm25_hit_index is None and chunk["bm25_hit"]:
            chunk_bm25_hit_index = i
        result.append(chunk)

    if chunk_bm25_hit_index is None:
        return None

    nr_negatives = min(nr_negatives, len(result))

    if nr_negatives == 1:
        left = result[chunk_bm25_hit_index:chunk_bm25_hit_index + 1]
        right = []
    elif chunk_bm25_hit_index < nr_negatives:
        left = result[:chunk_bm25_hit_index]
        right = result[chunk_bm25_hit_index:nr_negatives]
    else:
        left = result[:chunk_bm25_hit_index]
        right = result[chunk_bm25_hit_index:]
        if len(left) < nr_negatives // 2:
            right = right[:nr_negatives // 2 + nr_negatives // 2 - len(left)]
        elif len(right) < nr_negatives // 2:
            left = left[-nr_negatives // 2 - nr_negatives // 2 + len(right):]
        else:
            left = left[-nr_negatives // 2:]
            right = right[:nr_negatives // 2]
    return left + right


def get_dpr_split_instances(
        cfg,
        split_data,
        meta_data,
        url__to__chunks__map__from_split_data,
        url__to__chunks__map__from_hard_negative=None,
        instance_id__to__hard_negative_urls__map=None,
        MAX_INSTANCES=250_000,
        HARD_DOC_NEGATIVES_AROUND_BM25_HIT=False,
):

    NR_NEGATIVE_PASSAGES = cfg.create_data.nr_negative_passage_samples
    NR_NEGATIVE_DOC_CTXT_PASSAGES = cfg.create_data.nr_negative_doc_ctxt_passages
    WORDS_CITATION_PREFIX = cfg.create_data.word_citation_prefix
    WORDS_CITATION_SUFFIX = cfg.create_data.word_citation_prefix

    url_from_train_data_keys = list(url__to__chunks__map__from_split_data.keys())

    instance_template = {
        "dataset": "wafer-kiltweb_100w_dpr",
        "question": "",
        "answers": list(),
        "positive_ctxs": list(),
        "negative_ctxs": list(),
        "negative_doc_ctxs": list(),
        "hard_negative_ctxs": list(),
        "hard_negative_doc_ctxs": list(),
    }

    with open(split_data) as f_in:

        out = list()
        for inst, line in enumerate(tqdm.tqdm(f_in)):

            jline = json.loads(line.strip())
            if jline["id"] not in meta_data["ids"]:
                continue

            # max main memory restriction
            if len(out) == MAX_INSTANCES:
                break

            train_instance = copy.deepcopy(instance_template)
            SEP_split = jline["input"].split("[SEP]")
            if len(SEP_split) != 3:
                continue
            title, section, ctxt = SEP_split
            before_cit, after_cit = ctxt.split("[CIT]")
            train_instance["question"] = " [SEP] ".join([
                " ".join(title.split()),
                " ".join(section.split()),
                " ".join(before_cit.split()[-WORDS_CITATION_PREFIX:] + [" [CIT] "] + after_cit.split()[:WORDS_CITATION_SUFFIX])
            ])

            #
            # create positive provenances
            #
            positive_provenance_urls = set()

            for i, output in enumerate(jline["output"]):
                if output["provenance"][0]["url"] in url__to__chunks__map__from_split_data:
                    train_instance["positive_ctxs"].extend(
                        url__to__chunks__map__from_split_data[output["provenance"][0]["url"]]
                    )
                    positive_provenance_urls.add(output["provenance"][0]["url"])

            negative_provenance_urls = set()

            #
            # create hard negative doc provenances
            #
            if cfg.create_data.create_hard_negatives and cfg.create_data.create_doc_negatives:
                for url in instance_id__to__hard_negative_urls__map[jline["id"]]:
                    if len(train_instance["hard_negative_doc_ctxs"]) >= NR_NEGATIVE_PASSAGES:
                        break
                    if url not in positive_provenance_urls:
                        res = _get_url_chunks_negatives(
                            url__to__chunks__map__from_hard_negative[url],
                            jline["id"],
                            NR_NEGATIVE_DOC_CTXT_PASSAGES,
                            None if HARD_DOC_NEGATIVES_AROUND_BM25_HIT else 0,
                        )
                        if res is not None:
                            train_instance["hard_negative_doc_ctxs"].extend(res)

                # fill up the hard negatives with more samples
                if len(train_instance["hard_negative_doc_ctxs"]) < NR_NEGATIVE_PASSAGES:
                    samples = numpy.random.randint(0, len(url_from_train_data_keys), NR_NEGATIVE_PASSAGES)
                    for sample in samples:
                        if len(train_instance["hard_negative_doc_ctxs"]) >= NR_NEGATIVE_PASSAGES:
                            break
                        negative_provenance_urls.add(url_from_train_data_keys[sample])
                        res = _get_url_chunks_negatives(
                            url__to__chunks__map__from_split_data[url_from_train_data_keys[sample]],
                            jline["id"],
                            NR_NEGATIVE_DOC_CTXT_PASSAGES,
                            sample % len(url__to__chunks__map__from_split_data[url_from_train_data_keys[sample]]) if HARD_DOC_NEGATIVES_AROUND_BM25_HIT else 0,
                        )
                        if res is not None:
                            train_instance["hard_negative_doc_ctxs"].extend(res)

            #
            # create hard negative passage provenances
            #
            if cfg.create_data.create_hard_negatives:
                for url in instance_id__to__hard_negative_urls__map[jline["id"]]:
                    if len(train_instance["hard_negative_ctxs"]) >= NR_NEGATIVE_PASSAGES:
                        break
                    if url not in positive_provenance_urls:
                        res = _get_url_chunks_negatives(
                            url__to__chunks__map__from_hard_negative[url],
                            jline["id"],
                            1,
                            None if HARD_DOC_NEGATIVES_AROUND_BM25_HIT else 0,
                        )
                        if res is not None:
                            train_instance["hard_negative_ctxs"].extend(res)

                # fill up the hard negatives with more samples
                if len(train_instance["hard_negative_ctxs"]) < NR_NEGATIVE_PASSAGES:
                    samples = numpy.random.randint(0, len(url_from_train_data_keys), NR_NEGATIVE_PASSAGES)
                    for sample in samples:
                        if len(train_instance["hard_negative_ctxs"]) >= NR_NEGATIVE_PASSAGES:
                            break
                        negative_provenance_urls.add(url_from_train_data_keys[sample])
                        res = _get_url_chunks_negatives(
                            url__to__chunks__map__from_split_data[url_from_train_data_keys[sample]],
                            jline["id"],
                            1,
                            sample % len(url__to__chunks__map__from_split_data[url_from_train_data_keys[sample]])
                        )
                        if res is not None:
                            train_instance["hard_negative_ctxs"].extend(res)

            #
            # create random negative doc provenances
            #
            if cfg.create_data.create_doc_negatives:
                samples = numpy.random.randint(0, len(url_from_train_data_keys), NR_NEGATIVE_PASSAGES)
                for sample in samples:
                    if len(train_instance["negative_doc_ctxs"]) >= NR_NEGATIVE_PASSAGES:
                        break
                    negative_provenance_urls.add(url_from_train_data_keys[sample])
                    res = _get_url_chunks_negatives(
                        url__to__chunks__map__from_split_data[url_from_train_data_keys[sample]],
                        jline["id"],
                        NR_NEGATIVE_DOC_CTXT_PASSAGES,
                        sample % len(url__to__chunks__map__from_split_data[url_from_train_data_keys[sample]]) if HARD_DOC_NEGATIVES_AROUND_BM25_HIT else 0,

                    )
                    if res is not None:
                        train_instance["negative_doc_ctxs"].extend(res)


            #
            # create random negative provenances
            #
            samples = numpy.random.randint(0, len(url_from_train_data_keys), NR_NEGATIVE_PASSAGES)
            for sample in samples:
                if len(train_instance["negative_ctxs"]) >= NR_NEGATIVE_PASSAGES:
                    break
                negative_provenance_urls.add(url_from_train_data_keys[sample])
                res = _get_url_chunks_negatives(
                    url__to__chunks__map__from_split_data[url_from_train_data_keys[sample]],
                    jline["id"],
                    1,
                    sample % len(url__to__chunks__map__from_split_data[url_from_train_data_keys[sample]]) if HARD_DOC_NEGATIVES_AROUND_BM25_HIT else 0,
                )
                if res is not None:
                    train_instance["negative_ctxs"].extend(res)

            out.append(train_instance)

        return out


def create_index_chunks(
        OUT_DATA_PATH,
        tag,
):
    """
    :param OUT_DATA_PATH:
    :param tag:
    :return:
    """
    chunk_id_memory = set()

    def _write_chunks(csv_writer, split_data):
        for inst in tqdm.tqdm(split_data):
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
                                    ctxt["bm25_hit"],
                                ])

    with open(f"{OUT_DATA_PATH}wafer-kiltweb_{tag}.tsv", "w") as f_out:

        csv_writer = csv.writer(f_out, delimiter="\t")

        for split in [f"wafer-train-kiltweb_{tag}.json", "wafer-dev-kiltweb.json"]:
            if os.path.exists(f"{OUT_DATA_PATH}{split}"):
                with open(f"{OUT_DATA_PATH}{split}") as f_in:
                    _write_chunks(csv_writer, json.load(f_in))

def run(cfg):

    OUT_DATA_PATH = cfg.create_data.output_directory
    CHUNKS_GLOB = cfg.create_data.ccnet_chunks_glob
    CHUNK_ID_URL_MAP = cfg.create_data.chunk_id_url_map

    TRAIN_SPLIT_DATA_FILE = resolve_file(cfg, cfg.create_data.gold_train_file)
    RETRIEVED_TRAIN_SPLIT_DATA_FILE = resolve_file(cfg, cfg.create_data.retrieved_train_file, error_not_exists=False)

    EVAL_SPLIT_DATA_FILES = resolve_file(cfg, cfg.create_data.gold_valid_file)
    RETRIEVED_EVAL_SPLIT_DATA_FILES = resolve_file(cfg, cfg.create_data.retrieved_valid_file, error_not_exists=False)

    os.makedirs(OUT_DATA_PATH, exist_ok=True)

    if cfg.create_data.create_doc_negatives:
        with open(CHUNK_ID_URL_MAP, "rb") as f:
            chunk_id__to__url__map = pickle.load(f)

    #
    # Collect hard negatives from all splits (train, valid)
    #

    if cfg.create_data.create_hard_negatives:

        train__id__to__hard_negative_urls__map, train_count_hard_negative_urls_dict, chunk_id__to__train_id__map = get_instance_id__to__hard_negative_urls__map(
            RETRIEVED_TRAIN_SPLIT_DATA_FILE,
            cfg,
        )

        valid__id__to__hard_negative_urls__map, valid_count_hard_negative_urls_dict, chunk_id__to__valid_id__map = get_instance_id__to__hard_negative_urls__map(
            RETRIEVED_EVAL_SPLIT_DATA_FILES,
            cfg,
        )

        all_count_hard_negative_urls_dict = dict()
        for d in [
            train_count_hard_negative_urls_dict,
            valid_count_hard_negative_urls_dict,
        ]:
            all_count_hard_negative_urls_dict.update(d)

        all_chunk_id__to__instance_id__map = dict()
        for d in [
            chunk_id__to__train_id__map,
            chunk_id__to__valid_id__map,
        ]:
            all_chunk_id__to__instance_id__map.update(d)

        ignore_urls = set([k for k, v in all_count_hard_negative_urls_dict.items() if v > 13])

        all__url__to__chunks__map__from_hard_negatives, all_urls = get_url__to__chunks__map__from_hard_negatives(
            chunk_id__to__url__map,
            ignore_urls,
            all_count_hard_negative_urls_dict,
            all_chunk_id__to__instance_id__map,
            CHUNKS_GLOB,
        )

    else:

        all__url__to__chunks__map__from_hard_negatives=None
        train__id__to__hard_negative_urls__map=None
        valid__id__to__hard_negative_urls__map=None

    url__to__chunks__map__from_train_split, train_meta_data = get_url__to__passages_from_split_data(
        TRAIN_SPLIT_DATA_FILE,
    )

    dpr_train_instances = get_dpr_split_instances(
        cfg,
        TRAIN_SPLIT_DATA_FILE,
        train_meta_data,
        url__to__chunks__map__from_train_split,
        all__url__to__chunks__map__from_hard_negatives,
        train__id__to__hard_negative_urls__map,
    )

    del url__to__chunks__map__from_train_split
    del train__id__to__hard_negative_urls__map

    train_out_file_name_prefix = OUT_DATA_PATH + "wafer-train-kiltweb"

    with open(f"{train_out_file_name_prefix}_010k.jsonl", "w") as f_out_010:
        with open(f"{train_out_file_name_prefix}_maxk.jsonl", "w") as f_out_max:
            for i in range(len(dpr_train_instances)):
                if i < 10_000:
                    f_out_010.writelines(json.dumps(dpr_train_instances[i]) + "\n")
                f_out_max.writelines(json.dumps(dpr_train_instances[i]) + "\n")

    url__to__chunks__map__from_valid_split, valid_meta_data = get_url__to__passages_from_split_data(
        EVAL_SPLIT_DATA_FILES,
    )

    dpr_valid_instances = get_dpr_split_instances(
        cfg,
        EVAL_SPLIT_DATA_FILES,
        valid_meta_data,
        url__to__chunks__map__from_valid_split,
        all__url__to__chunks__map__from_hard_negatives,
        valid__id__to__hard_negative_urls__map,
    )

    valid_out_file_name_prefix = OUT_DATA_PATH + "wafer-dev-kiltweb"
    with open(f"{valid_out_file_name_prefix}.jsonl", "w") as f_out:
        for i in range(len(dpr_valid_instances)):
            f_out.writelines(json.dumps(dpr_valid_instances[i]) + "\n")


@hydra.main(config_path="../../conf", config_name="create_data/reproduce_dpr_training")
def main(cfg):

    if cfg.create_data.output_directory is None or len(cfg.create_data.output_directory) == 0:
        print(f"'output_directory' was not set. Please run this script with {__file__} output_directory=PATH_TO_OUTPUT_DIR")
        exit()

    run(cfg)

    for jsonl_file in glob.glob(f"{cfg.create_data.output_directory}/*jsonl"):
        jlines = list()
        with open(jsonl_file) as f_in:
            for line in f_in:
                jlines.append(json.loads(line))
            with open(jsonl_file[:-1], "w") as f_out:
                f_out.writelines(json.dumps(jlines, indent=4))

    for tag in "010k", "200k":
        create_index_chunks(
            cfg.create_data.output_directory,
            tag,
        )

if __name__ == "__main__":
    main()

