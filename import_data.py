
import cv2
import os
import pickle as pkl
PATH_TO_HERLEV = 'C:/Users/Fixo casaa/Desktop/smear2005/New database pictures'
PATH_TO_CERVIX93 = 'C:/Users/joao/Desktop/Projecto tese e tp/Trabalhos Praticos/cytology_dataset-master/dataset'
PATH_TO_PICKLED_DATA ='C:/Users/Fixo casaa/PycharmProjects/Trabalhos_praticos/data'


def save_data(path, cervic_or_herlev):  ## dataset (0,1) where 0=cervix93 and 1=Herlev
    folder_list = []
    image_list = []
    gt_list = []

    dir_list = os.listdir(path)
    for each_folder in dir_list:
        folder_name = os.path.split(each_folder)
        folder_list.append(folder_name[-1])
    #print(folder_list)
    if cervic_or_herlev == 1:  ##  ['carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic', 'normal_columnar', 'normal_intermediate', 'normal_superficiel', 'severe_dysplastic']
        for each_img_folder in folder_list:
            path_to_images = os.path.join(path, each_img_folder)
            image_id = os.listdir(path_to_images)
            img_id_clean = []
            for each_str in image_id:

                [temp, garbage] = each_str.split('.')
                if temp != 'Thumbs':
                    img_id_clean.append(temp)
            #print(img_id_clean)
            remove = 'd'
            imgs_gt = []
            imgs_to_segment = []
            for each_element in img_id_clean:
                if each_element[-1] == remove:
                    imgs_gt.append(each_element)
                else:
                    imgs_to_segment.append(each_element)
            # imgs_to_segment_bool = img_id_clean[:][-1] != remove
            # imgs_gt_bool = img_id_clean[-1] == remove
            # imgs_to_segment = img_id_clean[imgs_to_segment_bool]
            # imgs_gt = img_id_clean[imgs_gt_bool]
            for each_item_s in imgs_to_segment:
                img = cv2.imread(os.path.join(path_to_images, each_item_s+'.bmp'), cv2.IMREAD_COLOR)
                #print(img.shape)
                image_list.append([img, each_item_s+';'+each_img_folder])

            for each_item_gt in imgs_gt:
                img = cv2.imread(os.path.join(path_to_images, each_item_gt+'.bmp'), cv2.IMREAD_COLOR)
                #print(img.shape)
                gt_list.append([img, each_item_gt+':'+each_img_folder])
        save_path_imgs = os.path.join(PATH_TO_PICKLED_DATA,'herlev_imgs.pickle')
        save_path_gt = os.path.join(PATH_TO_PICKLED_DATA, 'herlev_gt.pickle')

        with open(save_path_imgs, 'wb') as f_img:
            pkl.dump(image_list,f_img)

        with open(save_path_gt,'wb') as f_gt:
            pkl.dump(gt_list,f_gt)
        #return image_list,gt_list   ## format => list[image][name;folder]
    else:
        i=0

def get_data(path):
    with open(path,'rb') as f:
        list = pkl.load(f)
    return list

# save_data(PATH_TO_HERLEV, 1)

i = 0
