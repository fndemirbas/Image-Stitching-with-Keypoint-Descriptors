import sys
import cv2
import os
from merge import merge
#GET PANOS FROM DIRECTORY
all_pano_in_folder = [name for name in os.listdir('pa_2_dataset') if os.path.isdir(os.path.join('pa_2_dataset', name)) ]
#FEATURE EXTRACTION METHOD TYPE

if(len(sys.argv)==1):
    method="SIFT"
else:
    method=sys.argv[1]

for pano in all_pano_in_folder:
    print("----------------------------")
    print("Worked Pano: "+str(pano))
    print("Feature Extranction Method: "+str(method))
    print("Start Creating Panora")
    images_in_pano = os.listdir(os.path.join('pa_2_dataset', pano))
    image_left = cv2.imread(os.path.join('pa_2_dataset', pano, images_in_pano[0]))
    image_right = cv2.imread(os.path.join('pa_2_dataset', pano, images_in_pano[1]))

    merged_img=None
    pairs = []
    print("Fist Pair Generating")
    for i, img_name in enumerate(images_in_pano):
        if (i == len(images_in_pano) - 1):
            break
        image_left = cv2.imread(os.path.join('pa_2_dataset', pano, images_in_pano[i]))
        image_right = cv2.imread(os.path.join('pa_2_dataset', pano, images_in_pano[i + 1]))
        err, merged_img = merge(image_left, image_right,method)
        err=0
        #cv2.imwrite( str(i) + ".png", merged_img)
        # Keep pairs
        pairs.append(merged_img)
        print("Pair" + str(i))

    # While create panorama image
    while len(pairs) != 1:
        print("----------SUBPAIR-----------")
        pairs_copy = []
        for i in range(len(pairs)):

            if (i == len(pairs) - 1):
                break
            image_left = pairs[i]
            image_right = pairs[i+1]
            err, merged_img = merge(image_left, image_right ,method)
            #If no match, don't append to pairs array
            pairs_copy.append(merged_img)
            print("Pair" + str(i))
        pairs = pairs_copy.copy()

    cv2.imwrite("Stitched_Panorama_Full_" + str(pano) + ".png", merged_img)
    if err == 1:
        #cv2.imwrite("Stitched_Panorama_Full_Fail" + str(pano) + ".png", merged_img)
        print("Error: There is not enough matches to create homography matrix. More than 4 good matches are needed.\n It is started to create next panorama...\n\n")
        continue
