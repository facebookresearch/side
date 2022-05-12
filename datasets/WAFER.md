# WAFER data

| split | size | file |
| ------------- | ------------- | ------------- |
| [train](http://dl.fbaipublicfiles.com/side/wafer-train.jsonl.tar.gz) | 3805958 | [wafer-train.jsonl.tar.gz](http://dl.fbaipublicfiles.com/side/wafer-train.jsonl.tar.gz) |
| [dev](http://dl.fbaipublicfiles.com/side/wafer-dev.jsonl.tar.gz) | 4545 | [wafer-dev.jsonl.tar.gz](http://dl.fbaipublicfiles.com/side/wafer-dev.jsonl.tar.gz) |
| [test](http://dl.fbaipublicfiles.com/side/wafer-test.jsonl.tar.gz) | 4568 | [wafer-test.jsonl.tar.gz](http://dl.fbaipublicfiles.com/side/wafer-test.jsonl.tar.gz) |
| [fail-dev](http://dl.fbaipublicfiles.com/side/wafer-fail-dev.jsonl.tar.gz) | 725 | [wafer-fail-dev.jsonl.tar.gz](http://dl.fbaipublicfiles.com/side/wafer-fail-dev.jsonl.tar.gz) |
| [fail-test](http://dl.fbaipublicfiles.com/side/wafer-fail-test.jsonl.tar.gz) | 730 | [wafer-fail-test.jsonl.tar.gz](http://dl.fbaipublicfiles.com/side/wafer-fail-test.jsonl.tar.gz) |


### Download script

```bash
python data/download_data.py --dest_dir data --dataset wafer
```

### Structure of each record

```python
{
'id': # unique id
'input': # in-context claim with [CIT] tag
'output': [ # list of valid urls for the citation
    {
    'answer': # a url or {Failed verification}
    'provenance': [
        {
            'url':  # *mandatory* 
            'chunk_id': # from the Sphere retrieval engine
            'title': # not provided, use the script to get this
            'text': # not provided, use the script to get this
        }
    ] 
    }
]
'meta': 
    {
        'wikipedia_id': # KILT wikipedia_id
        'wikipedia_title': # KILT wikipedia_title
        'wikipedia_section': # KILT wikipedia_section
        'cit_paragraph_id': # KILT cit_paragraph_id
        'cit_offset': # KILT cit_offset
        'sentences': [] # sentences before claim - sentences[-1] contains [CIT]
        'featured': # fatured flag, assume false if not present
    }
}
```