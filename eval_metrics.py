import numpy as np
import cv2
import pickle as pkl
from scipy.spatial.distance import directed_hausdorff

EVAL_INDEX = 0


def compare_expertvsmethod(nuc_and_cyto_mask,
                           corresp_gt):
    TP = np.count_nonzero(cv2.bitwise_and(nuc_and_cyto_mask, corresp_gt))
    TN = np.count_nonzero(cv2.bitwise_and(cv2.bitwise_not(nuc_and_cyto_mask), cv2.bitwise_not(corresp_gt)))
    FP = np.count_nonzero(cv2.bitwise_and(cv2.bitwise_not(corresp_gt), nuc_and_cyto_mask))
    FN = np.count_nonzero(cv2.bitwise_and(cv2.bitwise_not(nuc_and_cyto_mask), corresp_gt))

    haus = directed_hausdorff(nuc_and_cyto_mask, corresp_gt)



    return TP, TN, FP, FN,haus

with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/b_list_seg.pkl', 'rb') as fid:
    gt = pkl.load(fid)
# with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/nuc_masks.pkl', 'rb') as fid_n:
#     nuc = pkl.load(fid_n)
# with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/cyto_masks.pkl', 'rb') as fid_c:
#     cyto = pkl.load(fid_c)
# segemnt= []
# for idx, each in enumerate(cyto):
#
#     only_cyto = np.where(each != 255, each, 128)
#     out = cv2.bitwise_or(nuc[idx],only_cyto)
#     segemnt.append(out)
# with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/my_segment.pkl', 'wb')as seg_id:
#     pkl.dump(segemnt,seg_id)
#
TP = 0
TN = 0
FP = 0
FN = 0
haus_d = 0
with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/my_segment.pkl','rb')as seg_id:
    segemnt=pkl.load(seg_id)
for idx, each in enumerate(segemnt):
    # print(idx)
    # cv2.imshow('each',each)
    # cv2.imshow('gt',gt[idx])
    # cv2.waitKey()
    TP, TN, FP, FN, haus = compare_expertvsmethod(each, gt[idx])
    TP += TP
    TN += TN
    FP += FP
    FN += FN
    haus_d+=haus[0]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
zsi = (2 * TP) / (2 * TP + FP + FN)
## probability of error
print(str(precision)+'\n')
print(str(recall)+'\n')
print(str(zsi)+'\n')
print(str(haus_d/idx)+'\n')
print(idx)
#
#
#
#
# cv2.imshow('gt', gt[EVAL_INDEX])
# cv2.imshow('nuc', nuc[EVAL_INDEX])
# cv2.imshow('cyto', cyto[EVAL_INDEX])
# cv2.waitKey()
# i = 0



