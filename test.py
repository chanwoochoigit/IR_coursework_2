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

    aa = (N11/N) * log2((N*N11)/((N11+N10)*(N01+N11)))
    bb = (N01/N) * log2((N*N01)/((N01+N00)*(N01+N11)))
    cc = (N10/N) * log2((N*N10)/((N10+N11)*(N10+N00)))
    dd = (N00/N) * log2((N*N00)/((N00+N01)*(N10+N00)))

    return aa + bb + cc + dd

def calc_chi(N11, N10, N01, N00):
    return ((N11 + N10 + N01 + N00) * pow(((N11 * N00) - (N10 * N01)),2))/((N11+N01)*(N11+N10)*(N10+N00)*(N01+N00))

print(calc_mi(2, 1, 20754, 12728))
print(calc_chi(49, 27652, 141, 774106))