from collections import Counter
from random import randrange

nums = [3,3,3,3,2,2,2,2,1,1]
nums_2 = [0,1,0,0,1,0,0,0,0,1]
from math import log2, pow

def calc_dcg(nums):
    mark = 0
    for i, x in enumerate(nums, 1):
        if i == 1:
            mark += x
        else:
            mark += x / log2(i)
    print(mark)



def calc_mi(N11, N10, N01, N00):
    N = N11 + N10 + N01 + N00

    try:
        aa = (N11/N) * log2((N*N11)/((N11+N10)*(N01+N11)))
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

def calc_chi(N11, N10, N01, N00):
    return ((N11 + N10 + N01 + N00) * pow(((N11 * N00) - (N10 * N01)),2))/((N11+N01)*(N11+N10)*(N10+N00)*(N01+N00))

def test_rand():
    for i in range(1000):
        n = randrange(1,21)
        if n == 20:
            print(n)

def extract(string):
    ids = []
    aa = string.replace(' ','').replace('\"','').split('+')
    for a in aa:
        num = a.split('*')[1]
        ids.append(num)

    print(ids)

def test_counter():
    stuff = ['a', 'b', 'd', 'f', 'b', 'd', 'f', 'd', 'f']
    aa = Counter(stuff)
    print(aa['g'])

# # print(calc_mi(2, 0, 20754, 12728))
# # print(calc_chi(2, 0, 20754, 12728))
# # test_rand()
# sample_str = '0.132 * "3" + 0.107 * "85" + 0.076 * "662" + 0.067 * "438" + 0.039 * "654" + 0.033 * "651" + 0.033 * "161" +' \
#              ' 0.030 * "1707" + 0.023 * "1855" + 0.019 * "841"'
#
# extract(sample_str)
