import itertools
import re
import sys
from collections import defaultdict
import json
from nltk.stem import PorterStemmer
from math import log2
#my preprocessing module from coursework 1
class Preprocessor():

    def __init__(self):
        self.stopwords = self.get_stopwords()
        self.corpus = self.load_corpus()

    def create_corpus(self):
        with open('train_and_dev.tsv', 'r') as f:
            raw_text = f.readlines()

        corpus = self.init_nd_dict()

        counter = 0
        current_corpus = ''
        cp_list = ['ot', 'nt', 'quran']
        for line in raw_text:
            processed = self.preprocess(line)
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

    def get_stopwords(self):
        with open('stopwords.txt') as f:
            stop_words = f.read().split('\n')
        return stop_words

    def init_nd_dict(self):
        return defaultdict(lambda : defaultdict(dict))

    def unique_from_array(self, items):
        items_1d = list(itertools.chain.from_iterable(items))
        unique_dump = []
        [unique_dump.append(x) for x in items_1d if x not in unique_dump]

        return unique_dump

    def count_frequencies(self, data, unique_words):
        word_names = []
        word_counts = []
        check_count = 0

        for word in unique_words:
            count = 0
            word_names.append(word)
            for word_to_compare in data:
                if len(word_to_compare) == len(word):
                    if word_to_compare == word:
                        count += 1
            word_counts.append(count)
            check_count += 1
            if check_count % 100 == 1:
                print('counting word number ... {}/{}'.format(check_count, len(unique_words)))
        word_freq = {"Word": word_names, "Count": word_counts}

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

    #get counts to calculate mutual information
    def get_mi_counts(self, term, cls):
        classes = self.corpus.keys()

        #find "non-current" class
        c0 = [] # len(c0) is always 2
        for c in classes:
            if c != cls:
                c0.append(c)

        N11, N10, N01, N00 = sys.float_info.epsilon, sys.float_info.epsilon, sys.float_info.epsilon, sys.float_info.epsilon

        #investigate document in the given class
        for doc in self.corpus[cls].keys():
            curr_doc = self.corpus[cls][doc]
            if term in curr_doc:
                N11 += 1
            elif term not in curr_doc:
                N01 += 1

        #investigate documents in other classes
        for c in c0:
            for doc in self.corpus[c].keys():
                curr_doc = self.corpus[c][doc]
                if term in curr_doc:
                    N10 += 1
                elif term not in curr_doc:
                    N00 += 1

        return N11, N10, N01, N00

    #calculate mutual information given all 4 counts
    def calc_mi(self, term, cls):
        N11, N10, N01, N00 = self.get_mi_counts(term, cls)

        N = N11 + N10 + N01 + N00

        mi = (N11 / N) * log2((N * N11) / ((N11 + N10) * (N01 + N11))) + \
             (N01 / N) * log2((N * N01) / ((N01 + N00) * (N01 + N11))) + \
             (N10 / N) * log2((N * N10) / ((N10 + N11) * (N10 + N00))) + \
             (N00 / N) * log2((N * N00) / ((N00 + N01) * (N10 + N00)))

        return mi

    def run_mi_calculation(self):
        mi_result = self.init_nd_dict()

        counter = 1
        for cls in self.corpus.keys():

            for doc in self.corpus[cls]:
                print('class: {}/3---------------------------------------------------'.format(counter))
                print('calculating mutual information...{}/{}'.format(doc, len(self.corpus[cls].keys())))
                for word in self.corpus[cls][doc]:
                    mi = self.calc_mi(word, cls)
                    mi_result[word][cls] = mi
            counter += 1

        with open('mi.json', 'w') as f:
            json.dump(mi_result, f)

        return mi_result

    def sort_result(self):
        with open('mi.json', 'r') as f:
            mi = json.load(f)
        to_sort = self.init_nd_dict()
        for word in mi.keys():
            for corpus in mi[word]:
                score = mi[word][corpus]
                print(score)

p = Preprocessor()
# corp = p.create_corpus()
corp = p.load_corpus()
# print(corp)
print(len(corp['ot'].keys()) + len(corp['nt'].keys()) + len(corp['quran'].keys()))
# print(p.get_mi_counts(1, 3))
# p.run_mi_calculation()
p.sort_result()