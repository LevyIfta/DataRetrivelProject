import numpy as np
from inverted_index_proj import InvertedIndex, MultiFileReader
from handler import ReadFromGcp
from collections import defaultdict, Counter
import re
import math
import nltk
import time
from nltk.corpus import stopwords
import pandas as pd
from scipy.sparse import csr_matrix
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = stopwords_frozen.union(corpus_stopwords)

def tokenize(text):
    """
        This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

        Parameters:
        -----------
        text: string , represting the text to tokenize.

        Returns:;
        list of tokens (e.g., list of tokens).
        """
    global all_stopwords
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    return list_of_tokens


def query_expansion(tokens, model, n):
    expanded_query = []
    for word in tokens[0]:
        try:
            similar_words = model.wv.most_similar(word, topn=n)
            for w, _ in similar_words:
                expanded_query.append(w)
        except KeyError:
            # word not in vocabulary
            pass
    return tokens[0] + expanded_query


def preprocess(text):
    return [simple_preprocess(text)]


class SearchHandler:
    """
    Class that handles the searches over the indexes

    ...

    Attributes
    ----------
    handler : ReadFromGcp obj
    inverted_index : InvertedIndex object

    Methods
    -------
    search_body(q):
        Function that returns the best 100 documents
    """

    def __init__(self):
        self.handler = ReadFromGcp("ir_proj_205888886")
        self.inverted_index = {
            "body": self.handler.get_inverted_index(source_idx=f"postings_text/index.pkl",
                                                    dest_file=f"text_index.pkl"),
            "title": self.handler.get_inverted_index(source_idx=f"postings_title/index.pkl",
                                                     dest_file=f"title_index.pkl"),
            "titles_dict": self.handler.load_pickle_file(source=f"titles_dict.pkl", dest=f"titles_dict.pkl"),
            # "page_rank": self.handler.load_pickle_file(source=f"pageRank.pickle", dest=f"pageRank.pkl")
            "norm": self.handler.load_pickle_file_dict(source=f"doc_norms.pkl", dest=f"norm_dict.pkl"),
            # "idf": self.handler.load_pickle_file(source=f"idf_text.pkl", dest=f"idf_text.pkl"),
            "page_view": self.handler.load_pickle_file_dict(source=f"page_views.pkl", dest=f"page_views.pkl"),
            "doc_len": self.handler.load_pickle_file_dict(source=f"doc_len.pkl", dest=f"doc_len.pkl"),
            "anchor": self.handler.get_inverted_index(source_idx=f"postings_anchor_text/index.pkl",
                                                      dest_file=f"anchor_index.pkl")
            # "body_stem": self.handler.get_inverted_index(source_idx=f"postings_gcp_text_stemmed/index.pkl",
            #                                          dest_file=f"text_stemmed_index.pkl"),
        }

    def search(self, q):
        text = preprocess(q)
        model = Word2Vec(text, window=5, min_count=1, workers=4)
        expand_tokens = query_expansion(text, model, 1)
        print(expand_tokens)

    def search_body2(self, q):
        epsilon = 0.000001
        query = tokenize(q)
        idx_body = self.inverted_index["body"]
        #DL = self.inverted_index["body"].doc_count
        DL = len(self.inverted_index["doc_len"].keys())
        Q = generate_query_tfidf_vector(query, idx_body, DL)
        dic = defaultdict(list)
        index = "text"  # index for gcp bucket folder
        candidates = {}
        doc_id_len = self.inverted_index["doc_len"]
        print(Q)
        for w in query:
            pos_lst = self.handler.read_posting_list(idx_body, w, index)
            dic[w] = pos_lst

        for word, post in dic.items():
            for doc_id, tf in dic[word]:
                if doc_id_len[doc_id] > 0:
                    candidates[(doc_id, word)] = (tf / doc_id_len[doc_id]) * math.log(DL / idx_body.df[word]+epsilon)

        D = generate_document_tfidf_matrix(query, idx_body, candidates)
        print(D.head())
        cos_sin = cosine_similarity(D, Q)
        top_n = get_top_n(cos_sin, 5)

        res = []
        for doc,score in top_n:
            try:
                res.append((doc, self.inverted_index["titles_dict"][doc]))
            except:
                res.append(doc)

        return res

    def search_body(self, q):
        """
        This function returns the 100 documents.

        Parameters
        ----------
        q : str
            The sound the animal makes (default is None)

        Return
        -----
        result : List[Tuples(doc_id, title)]
        """
        # Tokenize the query
        #idf = self.inverted_index["idf"]
        epsilon = .0000001
        query = tokenize(q)
        dic = defaultdict(list)
        # Read inverted index from the bucket
        idx_body = self.inverted_index["body"].read_index('.', 'text_index')
        doc_count = len(self.inverted_index["doc_len"].keys())
        doc_len = self.inverted_index["doc_len"]
        Q = generate_query_tfidf_vector(query, idx_body, doc_count)
        doc_rankings = {}
        index = "text"
        # Make a dictionary with the term and its posting list
        cos_dict = {}
        for w in query:
            pos_lst = self.handler.read_posting_list(idx_body, w, index)
            dic[w] = pos_lst

        # Calculate tf_idf for each document
        tfidf_doc_dict = {}
        for word, post in dic.items():
            for doc_id, tf in dic[word]:
                tfidf = (tf / doc_len[doc_id]) * math.log((doc_count / idx_body.df[word]), 10)
                tfidf_doc_dict[(doc_id, word)] = tfidf
                #print(tfidf)

                # if doc_id not in cos_dict:
                #     cos_dict[doc_id] = np.dot(tfidf, Q)
                # else:
                #     cos_dict[doc_id] = cos_dict[doc_id] + np.dot(tfidf, Q[word])

        doc_unq = np.unique([doc_id for doc_id, w in tfidf_doc_dict.keys()])
        D = pd.DataFrame(columns=query, index=doc_unq)
        # D.index = unique_candidates
        # D.columns = index.term_total.keys()

        for key in tfidf_doc_dict:
            tfidf = tfidf_doc_dict[key]
            doc_id, term = key
            D.loc[doc_id, term] = tfidf

        D = D.fillna(0)
        doc_norm = self.inverted_index["norm"]
        result_dict = {}
        q_norm = np.linalg.norm(list(Q))
        for doc_id in D.index:
            if doc_id in doc_unq:
                cos_sim = np.dot(D.loc[doc_id], Q) / (doc_norm[doc_id]*q_norm)
                result_dict[doc_id] = cos_sim

        sorted_d = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_d = dict(sorted_d)
        sorted_keys = list(sorted_d.keys())
        res = []
        for key in sorted_keys:
            if key in self.inverted_index["titles_dict"].keys():
                res.append((key, self.inverted_index["titles_dict"][key]))
        return res[:5]

    def search_title(self, q):
        """
        This function returns the ALL the documents who have one of the query words in thier title.

        Parameters
        ----------
        q : str
            q as a query to search for,
            ex: The sound the animal makes (default is None)

        Return
        -----
        result : List[Tuples(doc_id, title)]
        """
        query = tokenize(q)
        index = "title"
        # Read inverted index from the bucket
        idx_title = self.inverted_index["title"].read_index('.', 'title_index')
        dic = defaultdict(list)
        title_ranking = {}

        for w in query:
            dic[w] = self.handler.read_posting_list(idx_title, w, index)

        for key, val in dic.items():
            for doc_id, count in val:
                if doc_id not in title_ranking.keys():
                    title_ranking[doc_id] = 0
                title_ranking[doc_id] = title_ranking[doc_id] + 1

        sorted_d = dict(sorted(title_ranking.items(), key=lambda item: item[1], reverse=True))
        sorted_keys = list(sorted_d.keys())

        res = []
        for key in sorted_keys:
            res.append((key, self.inverted_index["titles_dict"].get(key,0)))
        return res

    def search_anchor(self, q):
        """
        This function returns the 100 documents.

        Parameters
        ----------
        q : str
            The sound the animal makes (default is None)

        Return
        -----
        result : List[Tuples(doc_id, title)]
        """
        query = tokenize(q)
        index = "anchor_text"
        dic = defaultdict(list)
        # Read inverted index from the bucket
        idx_anchor = self.inverted_index["anchor"].read_index('.', 'anchor_index')

        dic = defaultdict(list)
        anchor_ranking = {}
        for w in query:
            dic[w] = self.handler.read_posting_list(idx_anchor, w, index)

        for key, val in dic.items():
            for doc_id, count in val:
                if doc_id not in anchor_ranking.keys():
                    anchor_ranking[doc_id] = 0
                anchor_ranking[doc_id] = anchor_ranking[doc_id] + 1

        sorted_d = dict(sorted(anchor_ranking.items(), key=lambda item: item[1], reverse=True))
        sorted_keys = list(sorted_d.keys())
        res = []
        for key in sorted_keys:
            res.append((key, self.inverted_index["titles_dict"].get(key, 0)))
        return res


    def get_page_rank(self, docs):
        res = []

        for doc in docs:
            res.append(self.inverted_index["page_rank"][doc])
        return res

    def get_page_view(self, docs):
        res = []
        for doc in docs:
            res.append(self.inverted_index["page_view"][doc])
        return res


def generate_query_tfidf_vector(query_to_search, index, DL):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """
    epsilon = .0000001
    total_vocab_size = len(query_to_search)
    Q = np.zeros((total_vocab_size))
    term_vector = query_to_search
    counter = Counter(query_to_search)
    dic = {}
    for token in np.unique(query_to_search):
        tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
        df = index.df[token]
        idf = math.log((DL) / (df + epsilon), 10)  # smoothing
        try:
            ind = term_vector.index(token)
            Q[ind] = tf * idf
            dic[token] = tf * idf
        except Exception as e:
            print("In generate query tfidf", e)

    return Q


def get_candidate_documents_and_scores(query_to_search, index, words, pls, DL):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """

    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq / DL[str(doc_id)]) * math.log(len(DL) / index.df[term], 10)) for
                               doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, candidates):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.


    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(index.term_total)
    candidates_scores = candidates
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    #D = csr_matrix(total_vocab_size, len(unique_candidates))
    D = pd.DataFrame(columns=query_to_search, index=unique_candidates)
    # D.index = unique_candidates
    # D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key

        D.loc[doc_id, term] = tfidf
    return D.fillna(0)


from pandas.core.frame import Axis


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """

    result_dict = {}
    q_norm = np.linalg.norm(Q)
    for doc_id in D.index:
        cos_sim = np.dot(D.loc[doc_id], Q) / (np.linalg.norm(D.loc[doc_id]) * q_norm)
        result_dict[doc_id] = cos_sim
    return result_dict


def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=False)[
           :N]
