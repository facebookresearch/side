import glob
import json

from flask import Flask
app = Flask(__name__)

pred_files_content_dict = dict()
eval_files_content_dict = dict()

def normalize(s):
    return ''.join(c for c in s.strip().lower() if c.isalnum())

def highlight(text, query, color="cyan"):
    query_toks = set([normalize(t) for t in query.split()])
    return ' '.join([f"<span style='background: {color}'>{t}</span>" if normalize(t) in query_toks else t for t in text.split()])


def get_pred_files_content(
        pred_file_index,
        pred_files,
        eval_file_index,
        eval_files,
        instance_index,
):
    if pred_file_index not in pred_files_content_dict:
        with open(pred_files[pred_file_index]) as f:
            pred_files_content_dict[pred_file_index] = f.readlines()
    if eval_file_index not in eval_files_content_dict:
        with open(eval_files[eval_file_index]) as f:
            eval_files_content_dict[eval_file_index] = f.readlines()
    return \
        json.loads(pred_files_content_dict[pred_file_index][instance_index]),\
        json.loads(eval_files_content_dict[eval_file_index][instance_index])

@app.route('/<pred_file_index>/<eval_file_index>/<instance_index>/<highlight_mode>')
def index(pred_file_index=0, eval_file_index=0, instance_index=0, highlight_mode=0,):

    pred_files = list(glob.glob("/checkpoint/fabiopetroni/WAI/Samuel/predictions/*/*jsonl")) + list(glob.glob("/checkpoint/fabiopetroni/WAI/Samuel/predictions/*/*/*jsonl"))
    eval_files = list(glob.glob("/checkpoint/fabiopetroni/KILT/datasets-web/*.jsonl"))

    pred_file_index, eval_file_index, instance_index, highlight_mode = [int(i) for i in (pred_file_index, eval_file_index, instance_index, highlight_mode)]
    pred_files_content, eval_files_content = get_pred_files_content(pred_file_index, pred_files, eval_file_index, eval_files, instance_index,)

    pred_files_cells = "<div style='background-color: #eee; width: 100%; height: 250px; border: 1px dotted black; overflow: scroll;'>" + "".join([
        f'<a style="background: {"cyan" if i==pred_file_index else "white"}" href="/{i}/{eval_file_index}/{instance_index}/{highlight_mode}">{n}</a></br>' for i, n in enumerate(pred_files)
    ]) + "</div>"

    eval_files_cells = "<div style='background-color: #eee; width: 100%; height: 250px; border: 1px dotted black; overflow: scroll;'>" + "".join([
        f'<a style="background: {"cyan" if i==eval_file_index else "white"}" href="/{pred_file_index}/{i}/{instance_index}/{highlight_mode}">{n}</a></br>' for i, n in enumerate(eval_files)
    ]) + "</div>"

    eval_url = eval_files_content["output"][0]["provenance"][0]["url"]
    pred_urls = [p["url"] == eval_url for p in pred_files_content["output"][0]["provenance"]]

    p_at_1 = pred_urls[0]
    hits_at_100 = pred_urls.count(True)

    def eval_highlight_mode():
        if highlight_mode == 0:
            return eval_files_content["input"]
        if highlight_mode == 1:
            return eval_files_content["meta"]["sentences"][-1]


    return '<html>' \
           '<body>' \
           '<table style="width: 100%; table-layout: fixed;">' \
           '' \
           '' \
           '<tr>' \
           '<td>' \
           f'<a href="/{pred_file_index}/{eval_file_index}/{max(instance_index, 0)}/0">HIGHLIGHT INPUT</a> <br/>' \
           f'<a href="/{pred_file_index}/{eval_file_index}/{max(instance_index, 0)}/1">HIGHLIGHT SENTENCE</a>' \
           '</td>' \
           '<td>' \
           '</td>' \
           '</tr>' \
           '' \
           '' \
           '<tr>' \
           '<td>' \
           f'<a href="/{pred_file_index}/{eval_file_index}/{max(instance_index - 1, 0)}/{highlight_mode}">BACK</a>' \
           '</td>' \
           '<td>' \
           f'<a href="/{pred_file_index}/{eval_file_index}/{instance_index + 1}/{highlight_mode}">NEXT</a>' \
           '</td>' \
           '</tr>' \
           '' \
           '' \
           '<tr><td>' + pred_files_cells + '</td><td>' + eval_files_cells + '</td></tr>' \
           '' \
           f'<tr><td><br\><br\></td><td></td></tr>' \
           '' \
           f'<tr><td>{eval_files_content["meta"]["sentences"][-1]}</td><td></td></tr>' \
           '' \
           f'<tr><td><br\><br\></td><td></td></tr>' \
           '' \
           f'<tr><td style="background: {"yellow" if p_at_1 else "white"}">Precision@1: {p_at_1} </td><td style="background: {"yellow" if hits_at_100>0 else "white"}">HITS@100: {hits_at_100}</td></tr>' \
           '' \
           f'<tr><td><br\><br\></td><td></td></tr>' \
           '' \
           '<tr">' \
           '<td style="width: 50%; overflow: scroll; vertical-align: top;">' \
           '<div style="width: 100%; height: 800px; overflow-y: scroll;">' \
           '' + ''.join([
            f'<div style="width: 90%; margin: 20px; background: {"yellow" if p["url"] == eval_url else "white"}"><a href="{p["url"]}">{p["wikipedia_title"] if "wikipedia_title" in p else p["url"]}</a><br/>{highlight(p["text"], eval_highlight_mode())}<br/>{ "Retriever: " + p["retriever_sys"] if "retriever_sys" in p else ""}</div>' for p in pred_files_content["output"][0]["provenance"][:100]
           ]) + '' \
           '</div>' \
           '</td>' \
           '<td style="width: 50%; max-height: 1000px; overflow: scroll; vertical-align: top;">' \
           '<div style="width: 100%; height: 800px; overflow-y: scroll;">' \
           '' + highlight(eval_files_content["input"], eval_files_content["meta"]["sentences"][-1]).replace("[CIT]", "<span style='background: LightSalmon'>[CIT]</span>") + ''\
           '<pre>' \
           '' + json.dumps(eval_files_content, indent=4) + ''\
           '</pre>' \
           '</div>' \
           '</td>' \
           '</tr>' \
           '</table>' \
           '</body>' \
           '</html>'
