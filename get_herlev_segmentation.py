import import_data
import cv2
import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

PATH_TO_EXPERT_HERLEV_SEGMENTATION = 'D:/Users/joao/PycharmProjects/TP/data/herlev_gt.pickle'
PATH_TO_HERLEV_IMGS = 'D:/Users/joao/PycharmProjects/TP/data/herlev_imgs.pickle'
EVAL_INDEX = 750
median_kernel_size = 7
##FCM
n_clusters = 7
## morph
disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
## clahe
clip_limit = 2
window = (8, 8)


def separate_into_channels(segmented_masks):
    b_list_seg = []
    g_list_seg = []
    r_list_seg = []
    for each_image in segmented_masks:
        b_temp, g_temp, r_temp = cv2.split(each_image)
        b_list_seg.append(b_temp)
        g_list_seg.append(g_temp)
        r_list_seg.append(r_temp)

    return b_list_seg, g_list_seg, r_list_seg


def get_image_histogram(gray_img):
    plt.hist(gray_img.ravel(), 256, [0, 255])
    plt.show()
    return 0


def rgb_to_gray(images_list):
    gray_img_list = []
    for each_image_og in images_list:
        gray_img_list.append(cv2.cvtColor(each_image_og, cv2.COLOR_BGR2GRAY))
    return gray_img_list


def apply_median_filter(list, kernel):
    out = []

    for each_image in list:
        temp = cv2.medianBlur(each_image, kernel)
        out.append(temp)
    return out


def apply_clahe(clip_limit, window, image_list):
    out = []
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=window)
    for each_image in image_list:
        out.append(clahe.apply(np.uint8(each_image)))
    return out


def get_corresp_center(u):
    out = np.zeros((7, len(u[0, :])), dtype=np.int)
    a = u.T
    for patch_idx, each_row in enumerate(a):
        idx = np.argmax(each_row, axis=0)
        out[idx, patch_idx] = 1
    return out


def apply_FCM(img, ncenters):
    ## divide image into patches
    patch_array = extract_patches_2d(img, (3, 3))

    patch_centers = []
    pidx_list = []
    # cv2.imshow('asd', img)
    # cv2.imshow('reb', np.uint8(rebuild))

    # cv2.waitKey()
    for p_idx, each_patch in enumerate(patch_array):
        patch_centers.append(each_patch[1, 1])
        pidx_list.append(np.int(p_idx))

    centers_array = np.reshape(np.asarray(patch_centers), (-1, len(patch_centers)))
    # centers_array = np.reshape(np.asarray(patch_centers), (-1,))

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(centers_array, ncenters, 2, error=0.005, maxiter=1000, init=None)
    # fcm_centers = np.reshape(cntr[0, :], (-1, 1))

    Tn = np.average(cntr) * 0.8
    Tc = np.average(cntr) * 1.2
    p_cluster_matching = get_corresp_center(u)
    matching_cv_patch = np.hstack((cntr, p_cluster_matching))
    sorted_cntr_patch = matching_cv_patch[matching_cv_patch[:, 0].argsort()]
    # pidx_list = np.c_[pidx_list]  # np.reshape(np.asarray(pidx_list), (-1, 1))
    # p_fcm_centers = np.hstack([pidx_list, np.uint8(cntr)])  ##stats[stats[:,-1].argsort()[::-1]]
    # p_fcm_centers = p_fcm_centers[p_fcm_centers[:, -1].argsort()[::-1]]
    labels = np.zeros_like(patch_array)
    for each_row in sorted_cntr_patch:
        center_v = each_row[0]
        idx_bool = np.delete(each_row, [0], axis=0)
        if center_v <= Tn:

            labels[idx_bool == 1] = np.ones((3, 3)) * 255
        elif Tn < center_v and center_v <= Tc:
            labels[idx_bool == 1] = np.ones((3, 3)) * 127
        else:
            labels[idx_bool == 1] = np.zeros((3, 3)) * 127

    pixel_labels = labels
    # for each_center in p_fcm_centers:
    #     idx = each_center[0]
    #     center = each_center[-1]
    #     if center <= Tn:
    #         patch_label = np.ones((3, 3)) * 255
    #         pixel_l = 2
    #     elif center <= Tc and center > Tn:
    #         patch_label = np.ones((3, 3)) * 127
    #         pixel_l = 1
    #     else:
    #         patch_label = np.zeros((3, 3))
    #         pixel_l = 0
    #     labels[idx] = patch_label
    #     pixel_labels[idx] = pixel_l
    patch_reb = []
    l_reb = []
    for idx_reb, each_path_reb in enumerate(patch_array):
        each_patch = labels[idx_reb]
        l_reb.append(pixel_labels[idx_reb])
        patch_reb.append(each_patch)
    h_reb, w_reb = np.shape(img)
    FCM_Clustered_Image = reconstruct_from_patches_2d(np.asarray(patch_reb), (h_reb, w_reb))
    labels_image = reconstruct_from_patches_2d(np.asarray(l_reb), (h_reb, w_reb))

    # cv2.imshow('FCM_Clustered_Image', FCM_Clustered_Image)
    # cv2.waitKey()
    return FCM_Clustered_Image, labels_image


def load_FCM_results():
    with open('D:/Users/joao/PycharmProjects/TP/data/fcm_results', 'rb') as load_id:
        out = import_data.pkl.load(load_id)
    return out


def load_FCM_labels():
    with open('D:/Users/joao/PycharmProjects/TP/data/fcm_labels', 'rb') as load_id:
        out = import_data.pkl.load(load_id)
    return out


def apply_morph(img_list, struct):
    out = []

    for each_image in img_list:
        each_image = np.uint8(each_image)
        temp_o = cv2.morphologyEx(each_image, cv2.MORPH_OPEN, struct)
        out.append(cv2.morphologyEx(temp_o, cv2.MORPH_CLOSE, struct))
    return out


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', labeled_img)
    # cv2.waitKey()


def segment_into_nucleus_and_cyto(image, name_and_label):
    nuc = np.zeros_like(np.uint8(image))
    cyto = np.zeros_like(np.uint8(image))
    nuc[image == 2] = 255
    cyto[image >= 1] = 255
    ## select only largest nuc and cyto
    select_only_largest(cyto)
    # select_only_largest(nuc)
    return 0


def select_only_largest(image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    imshow_components(np.uint8(labels))
    return 0


segmented_list = import_data.get_data(PATH_TO_EXPERT_HERLEV_SEGMENTATION)
cell_image_list = import_data.get_data(PATH_TO_HERLEV_IMGS)
images_list, name_and_label_imgs_og = zip(*cell_image_list)
segmented_masks, name_and_label_exp_segment = zip(*segmented_list)

cv2.imshow(name_and_label_exp_segment[EVAL_INDEX], segmented_masks[EVAL_INDEX])
cv2.imshow(name_and_label_imgs_og[EVAL_INDEX], images_list[EVAL_INDEX])
# cv2.waitKey()
b_list_seg, g_list_seg, r_list_seg = separate_into_channels(segmented_masks)

compare = np.hstack(
    [b_list_seg[EVAL_INDEX], g_list_seg[EVAL_INDEX], r_list_seg[EVAL_INDEX]])
cv2.imshow('check', compare)

gray_list = rgb_to_gray(images_list)

cv2.imshow('RGB2GRAY', gray_list[EVAL_INDEX])

gray_blurr = apply_median_filter(gray_list, median_kernel_size)
# get_image_histogram(gray_blurr[EVAL_INDEX])

cv2.imshow('median blurr ', gray_blurr[EVAL_INDEX])
gray_blurr_clahe = apply_clahe(clip_limit,window,gray_blurr)
# get_image_histogram(gray_blurr_clahe[EVAL_INDEX])
# fcm clustering
print('FCM clustering start')
FCM_list = []

FCM_labeled = []
for each_gray_blurr in gray_blurr:
    img_FCM, labels_img = apply_FCM(each_gray_blurr, n_clusters)
    FCM_labeled.append(labels_img)
    FCM_list.append(img_FCM)

with open('D:/Users/joao/PycharmProjects/TP/data/fcm_results', 'wb')as fcm_id:
    import_data.pkl.dump(FCM_list, fcm_id)
with open('D:/Users/joao/PycharmProjects/TP/data/fcm_labels','wb') as l_id:
    import_data.pkl.dump(FCM_labeled,l_id)
FCM_list = load_FCM_results()
FCM_labeled = load_FCM_labels()
cv2.imshow('after FCM', FCM_list[EVAL_INDEX])

m_list = apply_morph(FCM_list, disk)
m_l_list = apply_morph(FCM_labeled, disk)
cv2.imshow('after_morph', m_list[EVAL_INDEX])
# cv2.imshow('after_morph_labels', m_l_list[EVAL_INDEX] * 127)

# segment_into_nucleus_and_cyto(m_list[EVAL_INDEX], name_and_label_imgs_og[EVAL_INDEX])

cv2.waitKey()
