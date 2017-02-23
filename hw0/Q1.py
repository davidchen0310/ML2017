#!/bin/python

import sys
import numpy as np

def fillMatrix(filePath):
    mat = []
    with open(filePath) as f:
        for line in f:
            mat.append(line.split(','))
    return [[int(element) for element in row] for row in mat]

matA = fillMatrix(sys.argv[1])
matB = fillMatrix(sys.argv[2])

res = np.dot(matA, matB).tolist()[0]
res.sort()

with open('ans_one.txt', 'w') as f:
    for element in res:
        print(element, file=f)
