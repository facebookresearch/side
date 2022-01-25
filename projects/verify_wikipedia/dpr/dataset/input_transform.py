import collections
import json
import logging
from typing import List

import torch

from dpr.dataset.utils import normalize_passage

logger = logging.getLogger(__name__)


Passage = collections.namedtuple("Passage", ["text", "title"])

class QueryContextsRaw(object):
    query: str
    positive_passages: List[Passage]
    negative_passages: List[Passage]
    hard_negative_passages: List[Passage]


WaferPassage = collections.namedtuple("WaferPassage", ["text", "title", "url", "passage_id", "bm25_hit", "rank", ])

class WaferQueryContextsRawText(QueryContextsRaw):
    query: str
    positive_passages: List[WaferPassage]
    negative_passages: List[WaferPassage]
    negative_doc_passages: List[WaferPassage]
    hard_negative_passages: List[WaferPassage]
    hard_negative_doc_passages: List[WaferPassage]


class SchemaPreprocessor:

    def __init__(self, cfg, tensorizer):
        self.cfg = cfg
        self.tensorizer = tensorizer

    def preprocess_query(self, question: str, training: bool=False) -> torch.Tensor:
        raise NotImplementedError

    def preprocess_passage(self, passage: Passage, training: bool=False) -> torch.Tensor:
        raise NotImplementedError

    def passage_struct_from_dict(self, ctx: dict):
        raise NotImplementedError


class WaferPreprocessor(SchemaPreprocessor):

    def preprocess_query(self, query, training: bool=False):

        # extract title, section, text from the query string
        SEP_split = query.split("[SEP]")
        if len(SEP_split) == 1:
            # logger.warning(f"Text did not have three SEP sections: {len(SEP_split)}")
            title, section, text_cit = "", "", SEP_split[0]
        elif len(SEP_split) == 2:
            # logger.warning(f"Text did not have three SEP sections: {len(SEP_split)}")
            title, section, text_cit = SEP_split[0], "", SEP_split[1]
        elif len(SEP_split) == 3:
            title, section, text_cit = SEP_split
        CIT_split = text_cit.split("[CIT]")

        if len(CIT_split) == 1:
            logger.warning(f"Text did not have CIT section: {len(CIT_split)}")
            before_cit, after_cit = CIT_split[0], ""
        else:
            before_cit, after_cit = text_cit.split("[CIT]")

        # do some normalization and some hacks
        title_text = " ".join(title.split())
        section_text = " ".join(section.split())
        # double cit because BertTokenizer replaces last token with [SEP]
        # now only get tokens in front of citation
        before_cit_text = " ".join(before_cit.split())

        # get question transform type; default: only use context left of citation
        question_transform_type = None
        if not hasattr(self.cfg, "input_transform") or (self.cfg.input_transform.question.type is None or len(self.cfg.input_transform.question.type) == 0):
            question_transform_type = "only_before_cit"
        else:
            question_transform_type = self.cfg.input_transform.question.type

        fields = {
            "title": None,
            "section": None,
            "bc_text": None,
            "ac_text": None,
        }

        if question_transform_type == "only_before_cit" or question_transform_type == "only_before_cit+title":
            fields["title"] = title_text
            fields["bc_text"] = before_cit_text
        elif question_transform_type == "only_before_cit+dropout_title+section":
            if training:
                dropout_field = torch.rand(1)
                if dropout_field[0] < self.cfg.input_transform.question.dropout:
                    fields["title"] = self.tensorizer.get_mask_token()
                else:
                    fields["title"] = title_text
            else:
                fields["title"] = title_text
            fields["section"] = section_text
            fields["bc_text"] = before_cit_text
        elif question_transform_type == "only_before_cit+dropout_title":
            if training:
                dropout_field = torch.rand(1)
                if dropout_field[0] < self.cfg.input_transform.question.dropout:
                    fields["title"] = self.tensorizer.get_mask_token()
                else:
                    fields["title"] = title_text
            else:
                fields["title"] = title_text
            fields["bc_text"] = before_cit_text
        elif question_transform_type == "only_before_cit+dropout_title+dropout_section":
            if training:
                dropout_field = torch.rand(1)
                if dropout_field[0] < self.cfg.input_transform.question.dropout:
                    fields["section"] = self.tensorizer.get_mask_token()
                else:
                    fields["section"] = section_text
            else:
                fields["section"] = section_text
            fields["title"] = title_text
            fields["bc_text"] = before_cit_text
        elif question_transform_type == "only_before_cit+section" or question_transform_type == "only_before_cit+title+section" or question_transform_type == "only_before_cit+section+title":
            fields["title"] = title_text
            fields["section"] = section_text
            fields["bc_text"] = before_cit_text
        elif question_transform_type == "only_before_cit+no_title":
            fields["bc_text"] = before_cit_text
        elif question_transform_type == "only_before_cit_30w" or question_transform_type == "only_before_cit_30w+title":
            fields["title"] = title_text
            fields["bc_text"] = " ".join(list(before_cit_text.split())[-30:])
        elif question_transform_type == "only_before_cit_30w+no_title":
            fields["bc_text"] = " ".join(list(before_cit_text.split())[-30:])
        else:
            raise Exception(f"Unknown question_transform_type: {question_transform_type}")

        field_tensors = {
            "title": None,
            "section": None,
            "bc_text": None,
            "ac_text": None,
        }

        self.tensorizer.set_pad_to_max(False)

        for k in field_tensors.keys():
            if fields[k] is not None:
                field_tensors[k] = self.tensorizer.text_to_tensor(fields[k], apply_max_len=False, add_special_tokens=False)

        fields_max_perc_size = {
            "title": 0.2 ,
            "section": 0.2,
            "bc_text": None,
            "ac_text": 0.2,
        }
        for k in field_tensors.keys():
            if fields_max_perc_size[k]:
                fields_max_perc_size[k] = int(fields_max_perc_size[k] * self.tensorizer.max_length)

        SEP_LEN = 1
        CLS_LEN = 1

        for k in ["ac_text", "section", "title"]:
            if sum([len(t)+SEP_LEN for t in field_tensors.values() if t is not None])+CLS_LEN > self.tensorizer.max_length:
                if field_tensors[k] is not None and len(field_tensors[k])+1 > fields_max_perc_size[k]:
                    field_tensors[k] = field_tensors[k][:(fields_max_perc_size[k]-SEP_LEN)]

        if sum([len(t)+SEP_LEN for t in field_tensors.values() if t is not None])+CLS_LEN > self.tensorizer.max_length:
            blocked_size = sum([len(field_tensors[k])+SEP_LEN for k in ["ac_text", "section", "title"] if field_tensors[k] is not None])
            free_size = self.tensorizer.max_length - blocked_size - SEP_LEN - CLS_LEN
            field_tensors["bc_text"] = field_tensors["bc_text"][-free_size:]

        tags = {
            "[CLS]": torch.tensor([self.tensorizer.get_cls_id()]),
            "[SEP]": torch.tensor([self.tensorizer.get_sep_id()]),
            "[CIT]": self.tensorizer.text_to_tensor("[CIT]", apply_max_len=False, add_special_tokens=False)[0:1],
        }

        ordered_fields_and_tags = [
            ("title", tags["[SEP]"]),
            ("section", tags["[SEP]"]),
            ("bc_text", tags["[CIT]"]),
            ("ac_text", tags["[SEP]"]),
        ]

        tensor_list = list()
        tensor_list.append(tags["[CLS]"])
        for field_name, tag in ordered_fields_and_tags:
            if field_tensors[field_name] is not None:
                tensor_list.append(field_tensors[field_name])
                tensor_list.append(tag)

        pad_size = self.tensorizer.max_length - sum([len(t) for t in tensor_list])
        if pad_size > 0:
            tensor_list.append((torch.ones(pad_size, dtype=torch.int) * self.tensorizer.get_pad_id()).long())

        self.tensorizer.set_pad_to_max(True)
        assert sum([len(t) for t in tensor_list]) == self.tensorizer.max_length, f"{[len(t) for t in tensor_list]} {sum([len(t) for t in tensor_list])}!={self.tensorizer.max_length}"
        return torch.cat(tensor_list, dim=0).long()

    def preprocess_passage(self, ctx: WaferPassage, training: bool=False) -> torch.Tensor:

        try:
            text = json.loads(ctx.text)["contents"] # for BM25 pyserini which can have json packed into the text
        except:
            text = ctx.text

        if hasattr(self.cfg, "normalize_passage"):
            if self.cfg.normalize:
                text = normalize_passage(text)
        else:
            text = normalize_passage(text)

        # handle incomplete or missing configs
        if hasattr(self.cfg, "passage_transform_type"):
            if (self.cfg.input_transform.passage.type is None or len(self.cfg.input_transform.passage.type) == 0):
                passage_transform_type = "title+text"
            else:
                passage_transform_type = self.cfg.input_transform.passage.type
        else:
            passage_transform_type = "title+text"

        fields = {
            "title": None,
            "text": None,
            "url": None,
            "rank": None,
        }

        field_tensors = {
            "title": None,
            "text": None,
            "url": None,
            "rank": None,
        }

        if passage_transform_type == "text":
            fields["text"] = text
        elif passage_transform_type == "title+text":
            fields["title"] = ctx.title
            fields["text"] = text

        self.tensorizer.set_pad_to_max(False)

        # create token tensors
        for k in field_tensors.keys():
            if fields[k] is not None:
                field_tensors[k] = self.tensorizer.text_to_tensor(fields[k], apply_max_len=False, add_special_tokens=False)

        tags = {
            "[CLS]": torch.tensor([self.tensorizer.get_cls_id()]),
            "[SEP]": torch.tensor([self.tensorizer.get_sep_id()]),
        }

        ordered_fields_and_tags = [
            ("title", tags["[SEP]"]),
            ("text", tags["[SEP]"]),
            ("url", tags["[SEP]"]),
            ("rank", tags["[SEP]"]),
        ]

        tensor_list = list()
        tensor_list.append(tags["[CLS]"])
        for field_name, tag in ordered_fields_and_tags:
            if field_tensors[field_name] is not None:
                tensor_list.append(field_tensors[field_name])
                tensor_list.append(tag)

        pad_size = self.tensorizer.max_length - sum([len(t) for t in tensor_list])
        if pad_size > 0:
            tensor_list.append((torch.ones(pad_size, dtype=torch.int) * self.tensorizer.get_pad_id()).long())

        self.tensorizer.set_pad_to_max(True)
        return torch.cat(tensor_list, dim=0)[:self.tensorizer.max_length].long()

    def passage_struct_from_dict(self, ctx: dict):
        return WaferPassage(
            ctx["text"],
            ctx["title"],
            ctx["url"],
            ctx["passage_id"] if "passage_id" in ctx else None,
            ctx["bm25_hit"] if "bm25_hit" in ctx else ctx["system"] if "system" in ctx else ctx["system_hit"] if "system_hit" in ctx else None,
            ctx["rank"] if "rank" in ctx else "[RANK:0]",
        )
