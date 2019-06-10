import numpy as np
import cv2
import pickle as pkl
from skimage.feature.texture import greycoprops, greycomatrix

EVAL_INDEX = 0


def nuc_area(component):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(component)
    area = stats[1, -1]
    return area


def cyto_area(component):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(component)
    area = stats[1, -1]
    return area


def perimeter(component):
    contour, hierarchy = cv2.findContours(component, 1, 2)
    cnt = contour[0]
    perimeter = cv2.arcLength(cnt, True)
    return perimeter


def nuc_compactness(area, peri):
    out = (peri ** 2) / area
    return out


def elipse_axis(img):
    contour, hierarchy = cv2.findContours(img, 1, 2)
    cnt = contour[0]
    rect = cv2.minAreaRect(cnt)
    (x, y), (width, height), angle = rect

    major_axis = height / 2
    minor_axis = width / 2
    return major_axis, minor_axis


def nuc_aspect_ratio(img):
    contour, hierarchy = cv2.findContours(img, 1, 2)
    cnt = contour[0]
    rect = cv2.minAreaRect(cnt)

    (x, y), (width, height), angle = rect  # TODO PLAY AROUND AND FIND OUT HOW IT WORKS TO GET WIDTH AND HEIGHT

    return width / height


def nuc_homogenity(img):  # TODO ATENCAO AQUI A IMAGEM TEM DE SER GRAYLEVEL COM MASCARA
    glcm = greycomatrix(img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
    homo = greycoprops(glcm, 'homogeneity')
    #  https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html
    out = max(homo[0])
    return out


def nuc_to_cyto(area_n, area_c):
    return area_n / area_c


def cell_compactness(cyto_peri, cyto_area):
    return (cyto_peri ** 2) / cyto_area


def entirecellarea(img):
    contour, hierarchy = cv2.findContours(img, 1, 2)
    cnt = contour[0]
    rect = cv2.minAreaRect(cnt)
    (x, y), (width, height), angle = rect
    area = width * height
    return area


def applymask(img, mask):
    out = cv2.bitwise_and(img, mask)
    return out


with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/nuc_masks.pkl', 'rb') as fid_n:
    nuc = pkl.load(fid_n)
with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/cyto_masks.pkl', 'rb') as fid_c:
    cyto = pkl.load(fid_c)
with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/nuc_and_cyto.pkl', 'rb') as fid_cn:
    cyto_and_nuc = pkl.load(fid_cn)
with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/grayscale_imgs.pkl', 'rb') as fid_g:
    gray = pkl.load(fid_g)
features = []
for idx, each in enumerate(nuc):
    area_n = nuc_area(each)
    peri_n = perimeter(each)
    comp_n = nuc_compactness(area_n, peri_n)
    m_axis, min_axis = elipse_axis(each)
    a_ratio = nuc_aspect_ratio(each)
    masked = applymask(each, gray[idx])
    homo = nuc_homogenity(masked)
    area_c = cyto_area(cyto_and_nuc[idx])
    n_c = area_n / area_c
    peri_cell = perimeter(cyto_and_nuc[idx])
    comp_c = cell_compactness(peri_cell, area_c)
    area_cell = entirecellarea(cyto_and_nuc[idx])
    out = [area_n, comp_n, m_axis, min_axis, a_ratio, homo, n_c, comp_c, area_cell]
    features.append(out)
feat_arr = np.asarray(features)
normalized = np.zeros_like(feat_arr)
for all in range(0, len(out)):
    normalized[:, all] = feat_arr[:, all] / max(feat_arr[:, all])
    i = 0
with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/names.pkl','rb') as name_id:
    name_and_folder = pkl.load(name_id)
# names =[]# np.chararray((len(name_and_folder), 1))
labels = np.zeros((len(name_and_folder), 1),dtype=int)
for idx_sep, each_row in enumerate(name_and_folder):
    name, folder = str(each_row).split(';')
    # names.append(name)
    if folder == 'normal_superficiel' or folder == 'normal_intermediate' or folder == 'normal_columnar':
        labels[idx_sep] = 0

    if folder == 'light_dysplastic':
        labels[idx_sep] = 1
    if folder == 'moderate_dysplastic' or folder == 'severe_dysplastic':
        labels[idx_sep] = 2
    if folder == 'carcinoma_in_situ':
        labels[idx_sep] = 3

with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/labels.pkl', 'wb') as label_id:
    pkl.dump(labels, label_id)


with open('D:/Users/joao/PycharmProjects/Trabalhos_praticos/data/features.pkl', 'wb') as feat_id:
    pkl.dump(normalized, feat_id)
i = 0
