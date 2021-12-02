import itertools
import random
import re
import time
from collections import defaultdict
import json
import time
from collections import defaultdict
from math import log2
from random import randrange
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
import numpy as np
from collections import Counter
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from gensim.models import LdaModel
from nltk.stem import PorterStemmer
from math import log2
from scipy import sparse
#my preprocessing module from coursework 1
import pickle
from sklearn.model_selection import train_test_split


class Eval():

    def __init__(self, path):
        self.correct = self.get_corrects()
        self.true = self.get_rel_docs()
        self.result_path = path
        self.results = self.get_results()

    def get_results(self):
        with open(self.result_path, 'r') as f:
            return f.readlines()

    def get_corrects(self):
        with open('qrels.csv', 'r') as f:
            return f.readlines()

    def init_nd_dict(self):
        return defaultdict(lambda : defaultdict(dict))

    #get relative documents from qrels.csv, including relavance score
    def get_rel_docs(self):
        correct_dict = self.init_nd_dict()
        for line in self.correct:
            try:
                if 'query_id' not in line:
                    qid, rel_doc, relv = line.replace('\n','').split(',')
                    correct_dict[qid][rel_doc] = relv
            except:
                print('wrong input !-')
                continue
        return correct_dict

    def filter_results(self, cutoff):
        filtered_dict = self.init_nd_dict()

        for line in self.results:
            if 'system_number' and 'query_number' in line:
                # print('start of line - skipping')
                continue
            try:
                sys, q, doc, rk, sco = line.replace('\n','').split(',')
                if cutoff == 0: #if cutoff is set to 0 then just go without cutoff; i.e. no limit
                    filtered_dict[sys][q][doc] = sco
                else:
                    if int(rk) <= cutoff:
                        filtered_dict[sys][q][doc] = sco
            except:
                print('suspected wrong input!! please check around line 83')

        return filtered_dict

    def filter_results_dynamic(self):
        filtered_result = self.init_nd_dict()

        for line in self.results:
            if 'system_number' and 'query_number' in line:
                # print('start of line - skipping')
                continue
            try:
                sys, q, doc, rk, sco = line.replace('\n', '').split(',')
                r = len(self.true[q].keys())
                if int(rk) <= r:
                    filtered_result[sys][q][doc] = sco
            except:
                print('suspected wrong input!! please check around line 83')

        return filtered_result

    def get_mean_of_query(self, eval_dict):
        for sys in eval_dict.keys():
            mean = 0
            for query in eval_dict[sys]:
                mean += eval_dict[sys][query]
            eval_dict[sys]['mean'] = mean / len(eval_dict[sys].keys())

        return eval_dict

    def p_k(self, cutoff):
        start_time = time.time()
        filtered_result = self.filter_results(cutoff)

        p10_marks = self.init_nd_dict()

        for sys in filtered_result.keys():
            for q in filtered_result[sys]:
                mark = 0
                # print('======= sys: {} ========== query: {} ============'.format(sys, q))
                for doc in filtered_result[sys][q].keys():
                    if doc in self.true[q].keys():
                        # print(str(doc) +' | ' +str(self.true[q].keys()))
                        mark += 1

                p10_marks[sys][q] = mark / cutoff

        print('\n')
        print('{} seconds spent to calculate p@{}!'.format(round(time.time() - start_time, 5), cutoff))
        eval_result = self.get_mean_of_query(p10_marks)
        print(eval_result)
        return eval_result

    def r_k(self, cutoff):
        start_time = time.time()
        filtered_result = self.filter_results(cutoff)

        r50_marks = self.init_nd_dict()

        for sys in filtered_result.keys():
            for q in filtered_result[sys]:
                mark = 0
                # print('======= sys: {} ========== query: {} ============'.format(sys, q))
                for doc in filtered_result[sys][q].keys():
                    if doc in self.true[q].keys():
                        # print(str(doc) + ' | ' + str(self.true[q].keys()))
                        mark += 1

                r50_marks[sys][q] = mark / len(self.true[q].keys())

        print('\n')
        print('{} seconds spent to calculate r@{}!'.format(round(time.time() - start_time, 5), cutoff))
        eval_result = self.get_mean_of_query(r50_marks)
        print(eval_result)
        return eval_result

        """calculate precision at r: no. of relevant documents for query q"""
    def r_precision(self):
        start_time = time.time()
        filtered_result = self.filter_results_dynamic()

        rprc_marks = self.init_nd_dict()

        for sys in filtered_result.keys():
            for q in filtered_result[sys]:
                mark = 0
                # print('======= sys: {} ========== query: {} ============'.format(sys, q))
                for doc in filtered_result[sys][q].keys():
                    if str(doc) in self.true[q].keys():
                        # print(str(doc) +' | ' +str(self.true[q].keys()))
                        mark += 1

                rprc_marks[sys][q] = mark / len(self.true[q].keys())

        print('\n')
        print('{} seconds spent to calculate r-precision!'.format(round(time.time() - start_time, 5)))
        eval_result = self.get_mean_of_query(rprc_marks)
        print(eval_result)
        return eval_result


    def AP(self):
        start_time = time.time()
        results = self.filter_results(cutoff=0)
        ap_marks = self.init_nd_dict()

        for sys in results.keys():
            for q in results[sys].keys():
                ap = 0
                rel_counter = 1
                for docnos, doc in enumerate(results[sys][q].keys(), 1):
                    if doc in self.true[q].keys():
                        ap += rel_counter/docnos
                        rel_counter += 1
                ap_marks[sys][q] = ap / len(self.true[q].keys())

        print('\n')
        print('{} seconds spent to calculate AP!'.format(round(time.time() - start_time, 5)))
        eval_result = self.get_mean_of_query(ap_marks)
        print(eval_result)
        return eval_result

    def get_iDCG_k(self, cutoff):

        iDCG_k = self.init_nd_dict()  # query: ideal_dcg

        true_cp = self.true  # make a copy so the original true dict is not edited

        for q in true_cp.keys():
            # convert items with relevance of {} (empty due to predeclared nested dictionary) to 0
            for doc in true_cp[q].keys():
                if true_cp[q][doc] == {}:
                    true_cp[q][doc] = '0'

            #extend the ideal documents if the number of relevant docs is less than the cutoff
            while len(true_cp[q].keys()) < cutoff:
                true_cp[q]['padded_{}'.format(randrange(10,100))] = '0'

            ideal_sorted_docs = list(true_cp[q].items())

            #if no. of relevant docs is larger than cutoff, limit it to same as cutoff
            if len(true_cp[q].keys()) > cutoff:
                ideal_sorted_docs = ideal_sorted_docs[:cutoff]

            # sort document-relevance tuple by relevance
            # ideal_sorted_docs = list(sorted(true_cp[q].items(), key=lambda x: x[1], reverse=True))
            for k, item in enumerate(ideal_sorted_docs, 1):
                raw_gain = item[1]
                if k == 1:
                    iDCG_k[q] = int(raw_gain)
                elif k > 1:
                    iDCG_k[q] += int(raw_gain) / log2(k)
                else:
                    raise ValueError('wrong enumerator k value!')

        return iDCG_k

    def nDCG_k(self, cutoff):
        print('\n')
        start_time = time.time()
        filtered_results = self.filter_results(cutoff)

        iDCG_k = self.get_iDCG_k(cutoff)

        """""""""""""""""""""""""""""""""compute DCG (no normalisation)"""""""""""""""""""""""""""""""""
        DCG_k = self.init_nd_dict()
        for sys in filtered_results.keys():
            for q in filtered_results[sys].keys():
                doc_counter = 1
                for doc in filtered_results[sys][q]:
                    #retrieve relevance from the ideal search result; if doc doesn't exist because it's a bad search,
                    #set rel to 0
                    try:
                        rel = self.true[q][doc]
                        if rel == {}:
                            rel = 0
                    except:
                        rel = 0

                    if doc_counter == 1:
                        DCG_k[sys][q] = int(rel)
                    elif doc_counter > 1:
                        DCG_k[sys][q] += int(rel) / log2(doc_counter)
                    else:
                        raise ValueError('wrong enumerator i value!')
                    doc_counter += 1
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #now compute nDCG
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        for sys in DCG_k.keys():
            for q in DCG_k[sys].keys():
                print('__________________________________________')
                print('DCG - Sys {}, Query {}: {}'.format(sys, q, DCG_k[sys][q]))
                print('iDCG - Sys {}, Query {}: {}'.format(sys, q, iDCG_k[q]))
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

        nDCG_k = self.init_nd_dict()

        for sys in range(1, 7):
            for q in range(1, 11):
                dcg = DCG_k[str(sys)][str(q)]
                idcg = iDCG_k[str(q)]
                nDCG_k[str(sys)][str(q)] = dcg / idcg

        print('\n')
        print('{} seconds spent to calculate nDCG at {}'.format(round(time.time() - start_time, 5), cutoff))
        eval_result = self.get_mean_of_query(nDCG_k)
        print(eval_result)
        return eval_result

    def format_evaluation(self, p_10, r_50, rprc, AP, nDCG_10, nDCG_20):
        with open('ir_eval.csv', 'w') as f:
            f.write('system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20\n')

            for sys in ['1', '2', '3', '4', '5', '6']:
                for q in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'mean']:
                    f.write(str(sys) + ','+ str(q) + ',' + '{:.3f}'.format(round(p_10[sys][q], 3)) +
                                                     ',' + '{:.3f}'.format(round(r_50[sys][q], 3)) +
                                                     ',' + '{:.3f}'.format(round(rprc[sys][q], 3)) +
                                                     ',' + '{:.3f}'.format(round(AP[sys][q], 3)) +
                                                     ',' + '{:.3f}'.format(round(nDCG_10[sys][q], 3)) +
                                                     ',' + '{:.3f}'.format(round(nDCG_20[sys][q], 3)) +
                                                     '\n')

class Preprocessor():

    def __init__(self):
        self.stopwords = self.get_stopwords()

    def get_stopwords(self):
        with open('stopwords.txt') as f:
            stop_words = f.read().split('\n')
        return stop_words

    def unique_from_array(self, items):

        items_1d = list(itertools.chain.from_iterable(items.values()))
        vocab = {}
        for i, x in enumerate(items_1d):
            if x not in vocab.keys():
                vocab[x] = 0

        for i, k in enumerate(vocab.keys()):
            vocab[k] = i
        # using a rather unique structure to run faster
        # vocab[word] = word_index

        return vocab

    #convert word list to dictionary for speeding purposes
    def dictionify(self, items):
        word_dict = {}
        for i, word in enumerate(items):
            word_dict[i] = word
        return word_dict

    def encode_labels(self, labels):
        labels_encoded = []
        for l in labels:
            if l == 'ot':
                labels_encoded.append(0)
            elif l == 'nt':
                labels_encoded.append(1)
            elif l == 'quran':
                labels_encoded.append(2)
            else:
                raise ValueError('wrong corpus name!')
        return labels_encoded

    def create_count_matrix(self, docs, vocab, mode):
        count_mtx = sparse.dok_matrix((len(docs), len(vocab)), dtype='uint8')
        for i in docs.keys():
            if i % 3000 == 0:
                print('creating count matrix for {} SVM model ..... {}%'.format(mode, round(i / len(docs) * 100, 2)))
            count_dict = Counter(docs[i])
            for word in count_dict.keys():
                if mode == 'baseline':
                    try:
                        count_mtx[i, vocab[word]] = count_dict[word]
                    except:
                        continue
                elif mode == 'improved':
                    try:
                        count_mtx[i, vocab[word]] = count_dict[word] * 1000
                    except:
                        continue
                else:
                    raise ValueError('wrong mode choice!')
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

    def create_bigram_vectors(self, uni_vectors):
        bigram_vector = {}
        for vi, v in enumerate(uni_vectors):
            bv = []
            for i in range(len(v)-1):
                bv.append(str(v[i]+'_'+str(v[i+1])))
            bigram_vector[vi] = bv
        return bigram_vector

    def preprocess_baseline(self, document):
        # trim
        text_str = self.trim_text(document)

        # tokenise
        words_dup = self.tokenise(text_str)

        return words_dup

    #arbitrarily limit word length for  better accuracy (heuristic for lemmitisation)
    def limit_word_length(self, word_list, limit, offset):
        cut_text = []
        for word in word_list:
            if len(word) > limit:
                cut_text.append(word[:limit-offset])
            else:
                cut_text.append(word)

        return cut_text

    #preprocess 1-d list of text
    def preprocess(self, data_chunk):
        #trim
        text_str = self.trim_text(data_chunk)

        #tokenise
        words_dup = self.tokenise(text_str)

        #remove stop words
        # words_dup_nostop = self.remove_stopwords(words_dup)

        # """normalisation"""
        words_stemmed = self.stem_data(words_dup)

        # arbitrary cut to 4 chars if word length is longer than 5
        cut_off = self.limit_word_length(words_stemmed, 5, 1)

        #remove empty quotation marks ('')
        no_empties = self.remove_void(cut_off)

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
        self.raw_data = self.load_raw_data()
        self.raw_test_data = self.load_raw_test_data()

    def load_raw_data(self):
        with open('train_and_dev.tsv', 'r') as f:
            raw_text = f.readlines()

        return raw_text

    def load_raw_test_data(self):
        with open('test.tsv', 'r') as f:
            raw_text = f.readlines()

        return raw_text

    def shuffle_and_split(self, split, X, y):
        dataset = list(zip(X.todense(),y))  #zip the count matrix and labels
        random.shuffle(dataset)             #shuffle the cm-label tuples

        if split == 'train':    #if training set is given, split to training and validation
            X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1)
            X_train_sparse = sparse.dok_matrix(X_train)
            X_dev_sparse = sparse.dok_matrix(X_dev)
            return X_train_sparse, X_dev_sparse, y_train, y_dev

        elif split == 'test':
            splitted = [list(t) for t in zip(*dataset)] #unzip the list of tuples of [(dense_matrix, labels)]
            X_shuffled = splitted[0]
            y_shuffled = splitted[1]
            X_sparse = sparse.dok_matrix(np.concatenate(X_shuffled, axis=0)) #convert back to sparse matrix from dense
            return X_sparse, y_shuffled

    def collect_words_from_raw_text(self, mode, raw_text):
        p = Preprocessor()
        ####collect words from raw text#####################################################################
        docs = []
        labels = []

        for docid, line in enumerate(raw_text):
            if docid % 5000 == 0:
                print('building docs and preprocessing...{}%'.format(round(docid / len(raw_text) * 100, 2)))
            c, document = line.split('\t')
            if mode == 'baseline':
                docs.append(p.preprocess_baseline(document))
            elif mode == 'improved':
                docs.append(p.preprocess(document))
            else:
                raise ValueError('Wrong mode choice! It should be either baseline or improved.')
            labels.append(c.lower())
        ####################################################################################################
        return docs, labels

    #create vocabulary using the docs
    def create_vocab(self, docs):
        p = Preprocessor()
        vocab = p.unique_from_array(p.dictionify(docs)) # convert docs to be in dictionary form and create vocab

        return vocab

    def run_count_matrix_creator(self, mode, docs, vocab, labels):
        p = Preprocessor()

        docs = p.dictionify(docs)
        count_mtx = p.create_count_matrix(docs, vocab, mode)
        encoded_labels = p.encode_labels(labels)  # encode corpus labels; ot=0, nt=1, quran=2

        return count_mtx, encoded_labels

    def prepare_data(self, mode):
        raw_text = self.raw_data
        raw_test_text = self.raw_test_data

        docs, labels = self.collect_words_from_raw_text(mode, raw_text)
        test_docs, test_labels = self.collect_words_from_raw_text(mode, raw_test_text)

        vocab = self.create_vocab(docs) #create vocabulary using training data: test data doesn't effect the vocab

        count_mtx, encoded_labels = self.run_count_matrix_creator(mode, docs, vocab, labels)
        count_mtx_test, encoded_labels_test = self.run_count_matrix_creator(mode, test_docs, vocab, test_labels)

        X_train, X_dev, y_train, y_dev = self.shuffle_and_split('train', count_mtx, encoded_labels)
        X_test, y_test = self.shuffle_and_split('test', count_mtx_test, encoded_labels_test)

        #save shuffled and splitted data to disk
        with open('X_train_{}.pkl'.format(mode), 'wb') as f:
            pickle.dump(X_train, f)
        with open('X_test_{}.pkl'.format(mode), 'wb') as f:
            pickle.dump(X_test, f)
        with open('X_dev_{}.pkl'.format(mode), 'wb') as f:
            pickle.dump(X_dev, f)
        with open('y_train_{}.pkl'.format(mode), 'wb') as f:
            pickle.dump(y_train, f)
        with open('y_dev_{}.pkl'.format(mode), 'wb') as f:
            pickle.dump(y_dev, f)
        with open('y_test_{}.pkl'.format(mode), 'wb') as f:
            pickle.dump(y_test, f)

    def load_data(self, mode):
        with open('X_train_{}.pkl'.format(mode), 'rb') as f:
            X_train = pickle.load(f)
        with open('X_dev_{}.pkl'.format(mode), 'rb') as f:
            X_dev = pickle.load(f)
        with open('X_test_{}.pkl'.format(mode), 'rb') as f:
            X_test = pickle.load(f)
        with open('y_train_{}.pkl'.format(mode), 'rb') as f:
            y_train = pickle.load(f)
        with open('y_dev_{}.pkl'.format(mode), 'rb') as f:
            y_dev = pickle.load(f)
        with open('y_test_{}.pkl'.format(mode), 'rb') as f:
            y_test = pickle.load(f)

        return X_train, X_dev, X_test, y_train, y_dev, y_test

    def train_model(self, mode, classifier='svm'):
        if mode == 'baseline':
            c = 1000
            classifier = 'svm' #set baseline model to svm always
        elif mode == 'improved':
            c = 10
        else:
            raise ValueError('wrong mode to train SVM!!')

        X_train, X_dev, X_test, y_train, y_dev, y_test = self.load_data(mode)

        if classifier == 'linsvm':
            model = LinearSVC(C=c, max_iter=5000, verbose=True) #init sklearn.svm.LinearSVC for "improved" model
        elif classifier == 'nb':
            model = GaussianNB()
        elif classifier == 'svm':
            model = SVC(C=c, verbose=True) #init sklearn.svm.SVC
        else:
            raise ValueError('Wrong model choice! your current model: {}'.format(classifier))
        print("start training the {} model!".format(classifier))
        start_train = time.time()
        if classifier == 'nb':
            model.fit(X_train.todense(),y_train)
        else:
            model.fit(X_train,y_train)
        print('total training time: {} seconds'.format(time.time() - start_train))

        with open('{}_model_{}.pkl'.format(classifier, mode), 'wb') as f:
            pickle.dump(model, f)

        self.evaluate_predictions(mode, classifier)

    def load_svm_model(self, mode, classifier='svm'):
        with open('{}_model_{}.pkl'.format(classifier, mode), 'rb') as f:
            model = pickle.load(f)
        return model

        #required in the lab but not in cw2: only here to test the classification performance
        #not required in classification.csv
    def accuracy(self, y_true, y_pred):

        correct = 0
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                correct += 1

        return round(correct/ len(y_true),3)

    #initialise metrics dictinary for easier additions
    def init_metric_dict(self):
        a = Analyse()
        lookup = a.init_nd_dict()
        for i in range(3):
            lookup[i]['tp'] = 0
            lookup[i]['fp'] = 0
            lookup[i]['fn'] = 0

        return lookup

    def precision(self, y_true, y_pred):

        #initialise metrics dictionary
        lookup = self.init_metric_dict()

        for true, pred in zip(y_true, y_pred):
            if true == pred:
                lookup[pred]['tp'] += 1
            else:
                lookup[pred]['fp'] += 1

        precisions = {}
        for i in range(3):
            precisions[i] = round(lookup[i]['tp'] / (lookup[i]['tp'] + lookup[i]['fp']),3)

        precisions['macro'] = round((precisions[0] + precisions[1] + precisions[2])/3,3)

        return precisions

    def recall(self, y_true, y_pred):

        #initialise metrics dictionary
        lookup = self.init_metric_dict()

        for true, pred in zip(y_true, y_pred):
            if true == pred:
                lookup[true]['tp'] += 1
            else:
                lookup[true]['fn'] += 1

        recall = {}
        for i in range(3):
            recall[i] = round(lookup[i]['tp'] / (lookup[i]['tp'] + lookup[i]['fn']), 3)

        recall['macro'] = round((recall[0] + recall[1] + recall[2])/3,3)

        return recall

    def f1_score(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)

        f1 = {}
        for i in range(3):
            f1[i] = round( 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]),3)

        f1['macro'] = round((f1[0] + f1[1] + f1[2])/3,3)
        return f1

    def get_metrics_str(self, mode, split, y_true, y_pred):
        #OT = 0, NT = 1, Quran = 2

        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        f1 = self.f1_score(y_true, y_pred)

        metrics_string = ''
        metrics_string += mode + ',' + split+','                                           #add system and split
        metrics_string += str(precision[2]) + ',' + str(recall[2]) + ',' + str(f1[2]) + ','      #add p, r, f of Quran
        metrics_string += str(precision[0]) + ',' + str(recall[0]) + ',' + str(f1[0]) + ','      #add p, r, f of OT
        metrics_string += str(precision[1]) + ',' + str(recall[1]) + ',' + str(f1[1]) + ','      #add p, r, f of NT
        metrics_string += str(precision['macro']) + ',' + str(recall['macro']) + ','  + str(f1['macro'])

        return metrics_string

    def evaluate_predictions(self, mode, classifier='svm'):
        model = self.load_svm_model(mode, classifier)
        X_train, X_dev, X_test, y_train, y_dev, y_test = self.load_data(mode)
        if classifier == 'nb':
            y_train_pred = model.predict(X_train.todense())
            y_dev_pred = model.predict(X_dev.todense())
            y_test_pred = model.predict(X_test.todense())
        else:
            y_train_pred = model.predict(X_train)
            y_dev_pred = model.predict(X_dev)
            y_test_pred = model.predict(X_test)

        with open('classification.csv', 'a') as f:
            f.write('system,split,p-quran,r-quran,f-quran,p-ot,r-ot,f-ot,p-nt,r-nt,f-nt,p-macro,r-macro,f-macro\n')
            f.write(self.get_metrics_str(mode, 'train', y_train, y_train_pred) + '\n')
            f.write(self.get_metrics_str(mode, 'dev', y_dev, y_dev_pred) + '\n')
            f.write(self.get_metrics_str(mode, 'test', y_test, y_test_pred) + '\n')
            f.write('\n')

"""""""""""""""""""""""""""""""run and test the evaluation module"""""""""""""""""""""""""""""""
e = Eval('system_results.csv')
idx = e.true
# print(idx)
result = e.get_results()
p_10 = e.p_k(10)
r_50 = e.r_k(50)
r_prec = e.r_precision()
AP = e.AP()
nDCG_10 = e.nDCG_k(10)
nDCG_20 = e.nDCG_k(20)
e.format_evaluation(p_10, r_50, r_prec, AP, nDCG_10, nDCG_20)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""test and run text analysis"""""""""""""""""""""""""""""""""""
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
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""test and run text classification"""""""""""""""""""""""""""""""""
c = Classifier()
modes = ['baseline', 'improved']
m = 1
mode = modes[m]
# c.prepare_data(mode)
c.train_model(mode)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""