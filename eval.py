import time
from collections import defaultdict
from math import log2
from random import randrange


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