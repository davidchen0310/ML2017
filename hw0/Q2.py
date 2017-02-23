#!/bin/python

import sys
import numpy as np
from PIL import Image
from numpy import array

im_original = Image.open(sys.argv[1])
im_modified = Image.open(sys.argv[2])

im_res = Image.new('RGBA', im_original.size)

for x in range(im_original.size[0]):
    for y in range(im_original.size[1]):
        if (im_original.getpixel((x,y)) == im_modified.getpixel((x,y))):
            im_res.putpixel((x,y), (0,0,0,0))
        else:
            im_res.putpixel((x,y), im_modified.getpixel((x,y)))

im_res.save("ans_two.png")

#def imageToMatrix(imagePath):
#    im = Image.open(imagePath)
#    return array(im)
#
#mat_original = imageToMatrix("testdata/lena.png")
#mat_modified = imageToMatrix("testdata/lena_modified.png")
#
#res_mat = []
#for row_original, row_modified in zip(mat_original, mat_modified):
#    temp = []
#    for tup_original, tup_modified in zip(row_original, row_modified):
#        if (np.array_equal(tup_original, tup_modified)):
#            temp.append((0,0,0,0))
#        else:
#            temp.append(tup_modified)
#    res_mat.append(temp)
#
#res_image = Image.fromarray(array(res_mat))
#res_image.save("testdata/ans_lena.png")      
