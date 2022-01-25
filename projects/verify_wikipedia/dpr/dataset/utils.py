import glob
import os
from typing import List


def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text


def find_or_download_files(source_name) -> List[str]:
    if os.path.exists(source_name) or glob.glob(source_name):
        return glob.glob(source_name)
    # else:
    #     # try to use data downloader
    #     from dpr.data.download_data import download
    #     return download(source_name)


def get_file_len(filename):
    """
    From https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697

    :param filename:
    :return:
    """

    def _make_gen(reader):
        b = reader(2 ** 16)
        while b:
            yield b
            b = reader(2 ** 16)

    with open(filename, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
    return count


def resolve_file(cfg, file_name, error_not_exists=True):
    if file_name is None:
        if error_not_exists:
            print(f"File was 'None'.")
            exit()
        else:
            return None
    if not os.path.exists(file_name):
        if file_name in cfg.create_data.datasets:
            return cfg.create_data.datasets[file_name]['file']
        else:
            if error_not_exists:
                print(f"File {file_name} does not exist and is not an entry in datasets.")
                exit()
    return file_name