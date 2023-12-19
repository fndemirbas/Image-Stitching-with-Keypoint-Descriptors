# Image-Stitching-with-Keypoint-Descriptors
It is aimed to obtain a final panorama image that will combine the subimages provided using key point identification methods (SIFT/SURF and ORB) and include all the scenes in the subimages.

## IMPORTANT INFORMATION
Since the method I used to create a panorama image from subimages is a bit long, you can export the dataset from a small number of images.

## Requirement
- python3 (or higher)
- opencv 3 (or higher)

You will need to install some package using `pip`:
- numpy
- matplotlib

## Usage
$ python main.py <feature extraction method name(SIFT or ORB)>

### for example
$ python main.py SIFT
$ python main.py ORB
$ python main.py  // SIFT is selected manually


## Input format
The dataset folder must be in the same directory as main.py

## Output
The program will output:
- Showing every feature extraction, matched and Registration images by using matplotlib.
   If you don't want to see it. Comment lines 45, 50, 67, 94 in merge.py.
- A panoma stitched images `Stitched_Panorama_Full_{pano_name}.png`


## Environment
I test my code in Window11. It should work fine in these system.
