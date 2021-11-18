nums = [3,3,3,3,2,2,2,2,1,1]
nums_2 = [0,1,0,0,1,0,0,0,0,1]
from math import log2

def calc_dcg(nums):
    mark = 0
    for i, x in enumerate(nums, 1):
        if i == 1:
            mark += x
        else:
            mark += x / log2(i)
    print(mark)

calc_dcg(nums_2)