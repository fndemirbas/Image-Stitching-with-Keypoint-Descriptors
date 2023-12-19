import cv2
import numpy as np
from matplotlib import pyplot as plt
from homography import findHomographyMatrix
from merging import warpPerspective

def getSIFTFeatures(image):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp,des

def getORBFeatures(image):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

def merge(image_left, image_right, method):

    #Convert images to gray scale
    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

    #FEATURE EXTRACTION
    #ORB Descriptors
    #orb = cv2.SIFT_create()
    #kp1, des1 = orb.detectAndCompute(image_left_gray, None)
    #kp2, des2 = orb.detectAndCompute(image_right_gray, None)

    #FEATURE EXTRACTION
    kp1, des1 = None , None
    kp2, des2 = None , None
    if(method=="ORB"):
        #ORB Descriptors
        kp1, des1 = getORBFeatures(image_left_gray)
        kp2, des2 = getORBFeatures(image_right_gray)
    else:
        #SIFT Descriptors
        kp1, des1 = getSIFTFeatures(image_left_gray)
        kp2, des2 = getSIFTFeatures(image_right_gray)

    #PLOT KEY POINTS ON LEFT AND RIGHT IMAGE
    feature_extraction_for_left = cv2.drawKeypoints(image_left, kp1, outImage=None, color=(0,255,0), flags=0)
    plt.imshow(cv2.cvtColor(feature_extraction_for_left, cv2.COLOR_BGR2RGB))
    plt.title("left")
    #plt.show()

    feature_extraction_right_out = cv2.drawKeypoints(image_right, kp2, outImage=None,color=(0,255,0), flags=0)
    plt.imshow(cv2.cvtColor(feature_extraction_right_out, cv2.COLOR_BGR2RGB))
    plt.title("right")
    #plt.show()

    #Check des of images is empty
    if(des1 is None):
        return 1, image_right
    if (des2 is None):
        return 1, image_left

    # FEATURE MATCHING
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    #matches = sorted(matches, key = lambda x:x.distance)

    #Plot Image Matching
    feature_matching_out = cv2.drawMatchesKnn(image_left,kp1,image_right,kp2,matches[:50],None, flags=2)
    plt.imshow(cv2.cvtColor(feature_matching_out, cv2.COLOR_BGR2RGB))
    plt.title("Matched Image")
    #plt.show()

    #Select goog matches
    #good_matches = matches[:50]
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    #GET COORDINATES OF GOOD MATCHED POINTS ON LEFT AND RIGHT IMAGE
    good_matched_points_in_left_images = np.zeros((len(good_matches), 2))
    good_matched_points_in_right_images = np.zeros((len(good_matches), 2))
    for i, match in enumerate(good_matches):
        good_matched_points_in_left_images[i, :] = kp1[match.queryIdx].pt
        good_matched_points_in_right_images[i, :] = kp2[match.trainIdx].pt


    #FIND HOMOGRAPHY MATRIX
    if(len(good_matches) > 4):
        homography_matrix = findHomographyMatrix(good_matched_points_in_left_images, good_matched_points_in_right_images)
        homography_inv = np.linalg.inv(homography_matrix)

        #Merging Two Images by Transformation
        merged_img = warpPerspective(image_left, image_right, homography_inv)

        #SHOW MERGED IMAGE
        plt.imshow(cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB))
        plt.title("Image Registration")
        #plt.show()

        #No error and return merged image
        return 0, merged_img
    else:
        #Return error and left image
        return 1, image_left


