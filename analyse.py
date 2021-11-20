import itertools
import re
import sys
from collections import defaultdict
import json
from random import randrange

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from gensim.models import LdaModel
from nltk.stem import PorterStemmer
from math import log2
from scipy import sparse

#my preprocessing module from coursework 1
class Preprocessor():

    def __init__(self):
        self.stopwords = self.get_stopwords()

    def get_stopwords(self):
        with open('stopwords.txt') as f:
            stop_words = f.read().split('\n')
        return stop_words

    def unique_from_array(self, items):
        items_1d = list(itertools.chain.from_iterable(items))
        unique_dump = []
        [unique_dump.append(x) for x in items_1d if x not in unique_dump]
        vocab = {}
        for i, word in enumerate(unique_dump):
            vocab[i] = word

        return vocab

    def create_count_matrix(self, docs, vocab):
        count_mtx = sparse.dok_matrix((len(docs), len(vocab)), dtype=int)

        for v_ in vocab.keys():
            if v_ % 10 == 0:
                print('counting word number ... {}%'.format(round(v_ / len(vocab) * 100, 3)))
            voca = vocab[v_]
            for d_ in docs.keys():
                counter = 0
                for word in docs[d_]:
                    if voca == word:
                        counter += 1
                count_mtx[d_, v_] = counter

        #convert to coo and save as npz because dok save is not available yet
        #convert back to dok after loading
        sparse.save_npz('count_matrix.npz', count_mtx.tocoo())

        return count_mtx

    def trim_text(self, text):
        text_str = text.replace('\n', ' ').replace('\t',' ').replace('  ',' ')  # replace \n with a space, and if that creates a double space, replace it with a single space
        return text_str.lower()

    def tokenise(self, text_str):
        words = re.split('\W+', text_str)
        words_lower = []
        for word in words:
            words_lower.append(word.lower())
        return words_lower

    def remove_stopwords(self, words):
        stop_words = self.stopwords
        words_dup_nostop = []
        [words_dup_nostop.append(x) for x in words if x not in stop_words]
        return words_dup_nostop

    def stem_data(self, words_preprocessed):
        ps = PorterStemmer()
        words_stemmed = []
        for word in words_preprocessed:
            words_stemmed.append(ps.stem(word))
        return words_stemmed

    def remove_void(self, word_list):
        clean = []
        for word in word_list:
            if word != '':
                clean.append(word)

        return clean

    def preprocess_baseline(self, data_chunk):
        # trim
        text_str = self.trim_text(data_chunk)

        # tokenise
        words_dup = self.tokenise(text_str)

        return words_dup

    # def create_vocab_vector(self, vocab, words):


    #preprocess 1-d list of text
    def preprocess(self, data_chunk):
        #trim
        text_str = self.trim_text(data_chunk)

        #tokenise
        words_dup = self.tokenise(text_str)

        #remove stop words
        words_dup_nostop = self.remove_stopwords(words_dup)

        # """normalisation"""
        words_stemmed = self.stem_data(words_dup_nostop)

        #remove empty quotation marks ('')
        no_empties = self.remove_void(words_stemmed)

        return no_empties

    #preprocess 2-d list of text
    def preprocess_many(self, data_chunk_loads):
        processed_chunks_loads = []
        for data in data_chunk_loads:
            processed_chunks_loads.append(self.preprocess(data))

        return processed_chunks_loads

class Analyse():

    def __init__(self):
        self.corpus = self.load_corpus()
        self.p = Preprocessor()

    def init_nd_dict(self):
        return defaultdict(lambda : defaultdict(dict))

    def create_corpus(self):
        with open('train_and_dev.tsv', 'r') as f:
            raw_text = f.readlines()

        corpus = self.init_nd_dict()

        counter = 0
        current_corpus = ''
        cp_list = ['ot', 'nt', 'quran']
        for line in raw_text:
            processed = self.p.preprocess(line)
            head = processed[0]
            if current_corpus not in cp_list:
                current_corpus = head
            if current_corpus != head:
                current_corpus = head
                counter = 0
            corpus[current_corpus][counter] = processed[1:]
            counter += 1

        with open('corpus.json', 'w') as f:
            json.dump(corpus, f)

        return corpus

    def load_corpus(self):
        with open('corpus.json') as f:
            corpus = json.load(f)
        return corpus

    # get counts to calculate mutual information
    def get_Ns(self, term, cls):
        classes = self.corpus.keys()

        # find "non-current" class
        c0 = []  # len(c0) is always 2
        for c in classes:
            if c != cls:
                c0.append(c)

        N11, N10, N01, N00 = 0, 0, 0, 0

        # investigate document in the given class
        for doc in self.corpus[cls].keys():
            curr_doc = self.corpus[cls][doc]
            if term in curr_doc:
                N11 += 1
            elif term not in curr_doc:
                N01 += 1

        # investigate documents in other classes
        for c in c0:
            for doc in self.corpus[c].keys():
                curr_doc = self.corpus[c][doc]
                if term in curr_doc:
                    N10 += 1
                elif term not in curr_doc:
                    N00 += 1

        return N11, N10, N01, N00

    # calculate mutual information given all 4 counts
    def calc_mi(self, term, cls):
        N11, N10, N01, N00 = self.get_Ns(term, cls)

        N = N11 + N10 + N01 + N00

        try:
            aa = (N11 / N) * log2((N * N11) / ((N11 + N10) * (N01 + N11)))
        except:
            aa = 0
        try:
            bb = (N01 / N) * log2((N * N01) / ((N01 + N00) * (N01 + N11)))
        except:
            bb = 0
        try:
            cc = (N10 / N) * log2((N * N10) / ((N10 + N11) * (N10 + N00)))
        except:
            cc = 0
        try:
            dd = (N00 / N) * log2((N * N00) / ((N00 + N01) * (N10 + N00)))
        except:
            dd = 0

        return aa + bb + cc + dd

    def calc_chi(self, term, cls):
        N11, N10, N01, N00 = self.get_Ns(term, cls)
        return ((N11 + N10 + N01 + N00) * pow(((N11 * N00) - (N10 * N01)), 2)) / \
               ((N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00))

    # run mi or chi calculation
    def run_calculation(self, mode):

        result = self.init_nd_dict()

        counter = 1
        for cls in self.corpus.keys():

            for doc in self.corpus[cls]:
                print('class: {}/3---------------------------------------------------'.format(counter))
                print('calculating mutual information...{}/{}'.format(doc, len(self.corpus[cls].keys())))
                for word in self.corpus[cls][doc]:
                    if mode == 'mi':
                        score = self.calc_mi(word, cls)
                    elif mode == 'chi':
                        score = self.calc_chi(word, cls)
                    else:
                        raise ValueError('wrong calcluation mode entered! - choose mi or chi')
                    result[word][cls] = score
            counter += 1

        with open('{}.json'.format(mode), 'w') as f:
            json.dump(result, f)

        return result

    def sort_dict_by_value(self, dict_to_sort):
        return dict(sorted(dict_to_sort.items(), key=lambda item: item[1], reverse=True))

    def display_ranked_result(self, result_dict):
        for i, item in enumerate(result_dict.items()):
            term = item[0]
            score = item[1]
            print(term + ': ' + str(score))
            if i > 10:
                break

    def sort_result(self, mode):
        with open('{}.json'.format(mode), 'r') as f:
            to_display = json.load(f)
        to_sort = self.init_nd_dict()
        for word in to_display.keys():
            for corpus in to_display[word]:
                score = to_display[word][corpus]
                to_sort[corpus][word] = score

        sorted_ot = self.sort_dict_by_value(to_sort['ot'])
        sorted_nt = self.sort_dict_by_value(to_sort['nt'])
        sorted_qu = self.sort_dict_by_value(to_sort['quran'])

        self.display_ranked_result(sorted_ot)
        print('----------------------------')
        self.display_ranked_result(sorted_nt)
        print('----------------------------')
        self.display_ranked_result(sorted_qu)

    #helper function for get_lda_corpus
    # RETURNS: 2d list of documents based on self.corpus
    def get_all_docs(self):
        united_corpus = []

        # add the three corpus as one
        for cor in self.corpus.keys():
            for doc in self.corpus[cor].keys():
                united_corpus.append(self.corpus[cor][doc])

        return united_corpus

    def get_lda_corpus(self):
        # format the existing corpus "self.corpus" to fit in the gensim's LDA model.
        united_corpus = self.get_all_docs()
        corp_dictionary = Dictionary(united_corpus)
        corpus = [corp_dictionary.doc2bow(text) for text in united_corpus]

        return corpus

    def train_lda(self, k):
        # r = randrange(100)
        # print(r)
        lda = LdaModel(corpus=self.get_lda_corpus(), num_topics=k)

        # save lda model
        save_loc = datapath('lda_model')
        lda.save(save_loc)

    def load_lda(self):
        return LdaModel.load(datapath('lda_model'))

    def reverse_dict(self, dictionary):
        ks, vs = dictionary.keys(), dictionary.values()
        return dict(zip(vs,ks))

    def convert_list_of_tuple_to_dict(self, lot):
        dic = {}
        for item in lot:
            topic, prob = item
            dic[topic] = prob
        return dic

    def lda_calc_average_score(self):

        len_ot, len_nt, len_qu = len(self.corpus['ot'].keys()), len(self.corpus['nt'].keys()), len(self.corpus['quran'].keys())
        lda_result_dict = self.init_nd_dict()

        lda_distrib = self.load_lda().get_document_topics(self.get_lda_corpus())

        #add results for each corpus to get average score for each topic
        for i, line in enumerate(lda_distrib):
            if i % 1000 == 0:
                print('converting the result to a disposable form...{}/{}'.format(i, len(lda_distrib)))
            line_dict = self.convert_list_of_tuple_to_dict(line)
            if i < len_ot:
                lda_result_dict['ot'][i] = line_dict
            elif len_ot <= i < len_ot + len_nt:
                lda_result_dict['nt'][i] = line_dict
            elif len_ot + len_nt <= i:
                lda_result_dict['quran'][i] = line_dict

        #set probability to 0 if a topic probability does not appear
        for c in lda_result_dict.keys():
            for doc in lda_result_dict[c].keys():
                for topic in range(0, 20):
                    try:
                        if lda_result_dict[c][doc][topic] == {}:
                            lda_result_dict[c][doc][topic] = 0
                    except:
                        lda_result_dict[c][doc][topic] = 0
        avg_scores = self.init_nd_dict()

        #calculate average probability 1) sum up the values
        for c in lda_result_dict.keys():
            for doc in lda_result_dict[c].keys():
                for topic in lda_result_dict[c][doc].keys():
                    try:
                        avg_scores[c][topic] += lda_result_dict[c][doc][topic]
                    except:
                        avg_scores[c][topic] = lda_result_dict[c][doc][topic]

        #calculate average probability 2) average the values by the total number of documents in each corpus
        for c in avg_scores.keys():
            for topic in avg_scores[c].keys():
                avg_scores[c][topic] = avg_scores[c][topic] / len(lda_result_dict[c].keys())

        #sort each corpus by the probability of each topic candidate
        for c in avg_scores.keys():
            avg_scores[c] = {k: v for k, v in sorted(avg_scores[c].items(), key=lambda item: item[1], reverse=True)}

        with open('avg_score_dict.json', 'w') as f:
            json.dump(avg_scores, f)

    #extract token ides from a string returned from lda.print_topic()
    def extract_tokens_from_lda_str(self, lda_token_string):
        ids = {}

        #get token ID : word dictionary to retrieve words
        corp_dictionary = Dictionary(self.get_all_docs())
        word_to_id = self.reverse_dict(corp_dictionary.token2id)

        pns = lda_token_string.replace(' ', '').replace('\"', '').split('+')
        for prob_num in pns:
            prob, num = prob_num.split('*')
            ids[word_to_id[int(num)]] = prob

        ids_sorted = {k: v for k, v in sorted(ids.items(), key=lambda item: item[1], reverse=True)}

        return ids_sorted

    def find_top_tokens(self):
        with open('avg_score_dict.json', 'r') as f:
            avg_scores = json.load(f)

        ot_topic_best = list(avg_scores['ot'].keys())[0]
        nt_topic_best = list(avg_scores['nt'].keys())[0]
        qu_topic_best = list(avg_scores['quran'].keys())[0]

        print('ot: '+ot_topic_best)
        print('nt: '+nt_topic_best)
        print('quran: '+qu_topic_best)

        #find key tokens for each corpus

        lda_token_str_ot = self.load_lda().print_topic(int(ot_topic_best))
        power_words_ot = self.extract_tokens_from_lda_str(lda_token_str_ot)

        lda_token_str_nt = self.load_lda().print_topic(int(nt_topic_best))
        power_words_nt = self.extract_tokens_from_lda_str(lda_token_str_nt)

        lda_token_str_qu = self.load_lda().print_topic(int(qu_topic_best))
        power_words_qu = self.extract_tokens_from_lda_str(lda_token_str_qu)

        print(power_words_ot)
        print(power_words_nt)
        print(power_words_qu)

        return ot_topic_best, nt_topic_best, qu_topic_best

class Classifier():

    def __init__(self):
        pass

    def prepare_data(self):
        p = Preprocessor()

        with open('train_and_dev.tsv', 'r') as f:
            raw_text = f.readlines()

        docs = {}
        labels = []
        for docid, line in enumerate(raw_text):
            if docid % 5000 == 0:
                print('building docs and preprocessing...{}%'.format(round(docid / len(raw_text) * 100, 2)))
            c, text = line.split('\t')
            docs[docid] = p.preprocess_baseline(text)
            labels.append(c)

        vocab = p.unique_from_array(list(docs.values()))
        count_mtx = p.create_count_matrix(docs, vocab)

    def load_cm(self):
        # convert back to dok after loading
        coo = sparse.load_npz('count_matrix.npz')
        return coo.todok()

a = Analyse()
# corp = a.create_corpus()
# corp = a.load_corpus()
# print(len(corp['ot'].keys()) + len(corp['nt'].keys()) + len(corp['quran'].keys()))
# print(a.get_mi_counts(1, 3))
# a.run_calculation('mi')
# a.run_calculation('chi')
# a.sort_result('mi')
# a.sort_result('chi')
# a.train_lda(k=20)
# a.lda_calc_average_score()
# a.find_top_tokens()
c = Classifier()
# c.prepare_data()
print(c.load_cm())