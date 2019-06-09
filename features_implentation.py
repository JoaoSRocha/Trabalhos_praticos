import numpy as np
import cv2



def nuc_area(component):
    retval, labels, stats, centroids=cv2.connectedComponentsWithStats(component)
    area = labels[-1]
    return labels
def cyto_area(component):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(component)
    area = labels[-1]
    return labels
def perimeter(component):
    contour,hierarchy = cv2.findContours(component,1,2)
    cnt = contour[0]
    perimeter  = cv2.arcLength(cnt,True)
    return perimeter
def nuc_compactness(area,peri):
    out =(peri**2)/area
    return out
def elipse_axis(img):
    contour, hierarchy = cv2.findContours(img, 1, 2)
    cnt = contour[0]
    elipse=cv2.ellipse(cnt) #TODO PLAY AROUND AND FIND OUT HOW IT WORKS TO GET MINOR AND MAJOR AXIS
    return 0
def nuc_aspect_ratio(img):
    contour, hierarchy = cv2.findContours(img, 1, 2)
    cnt = contour[0]
    rect = cv2.minAreaRect(cnt)

    box = cv2.boxPoints(rect)#TODO PLAY AROUND AND FIND OUT HOW IT WORKS TO GET WIDTH AND HEIGHT

    box = np.int0(box)
    return 0
def homogenity(img):
    #TODO https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html
    return 0
def nuc_to_cyto(area_n,area_c):
    return area_n/area_c
def cell_compactness(cyto_peri,cyto_area):
    return (cyto_peri**2)/cyto_area