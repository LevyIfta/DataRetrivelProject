from inverted_index_proj import InvertedIndex, MultiFileReader
import struct

import os
import google
from google.cloud import storage
from contextlib import closing
import pickle as pkl
import json


class ReadFromGcp:
    def __init__(self, bucket_name):

        self.client = storage.Client()
        self.TUPLE_SIZE = 6
        self.TUPLE_SIZE_BODY = 8
        self.TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
        self.bucket = self.client.get_bucket(bucket_name)

    def get_pickle_from_gcp(self, source, dest):
        blob = self.bucket.get_blob(source)
        blob.download_to_filename(dest)

    def load_pickle_file(self, source, dest):
        if dest not in os.listdir("."):
            self.get_pickle_from_gcp(source, dest)
        with open(dest, "rb") as f:
            title_list = pkl.load(f)
            title_dict = {}
        for d in title_list:
            for k, v in d.items():
                title_dict[k] = v
        return title_dict

    def load_pickle_file_dict(self, source, dest):
        if dest not in os.listdir("."):
            self.get_pickle_from_gcp(source, dest)
        with open(dest, "rb") as f:
            title_list = pkl.load(f)
        return title_list

    def get_inverted_index(self, source_idx, dest_file):
        if dest_file not in os.listdir("."):
            blob = self.bucket.get_blob(source_idx)
            blob.download_to_filename(dest_file)
        return InvertedIndex().read_index(".", dest_file.split(".")[0])

    def read_posting_list(self, inverted, w, index):

        try:
            with closing(MultiFileReader()) as reader:
                locs = inverted.posting_locs[w]
                for loc in locs:

                    if loc[0] not in os.listdir("."):
                        blob = self.bucket.get_blob(f'postings_{index}/{loc[0]}')
                        filename = loc[0]
                        blob.download_to_filename(filename)
                posting_list = []

                b = reader.read(locs, inverted.df[w] * self.TUPLE_SIZE)
                for i in range(inverted.df[w]):
                    doc_id = int.from_bytes(b[i * self.TUPLE_SIZE:i * self.TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * self.TUPLE_SIZE + 4:(i + 1) * self.TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                return posting_list
        except IndexError:
            return []

    def load_json_file_dict(self, source, dest):
        if dest not in os.listdir("."):
            self.get_json_from_gcp(source, dest)
        with open(dest, "rb") as f:
            doc_id_list = json.load(f)
        return doc_id_list

    def get_json_from_gcp(self, source, dest):
        blob = self.bucket.get_blob(source)
        blob.download_to_filename(dest)
