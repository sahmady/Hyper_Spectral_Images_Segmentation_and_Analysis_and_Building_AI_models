import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import pandas as pd

VNIRFileSource = "../../SegmentedImages/SourceFolder/VNIR.txt"
imageType = "VNIR"
imagesArray = []
with open(VNIRFileSource, "r") as listOfImages:
    for line in listOfImages:
        imagesArray.append(line.strip())

shortenImgNameArray = np.zeros((len(imagesArray), 1))
intensityDataSet = np.zeros((len(imagesArray), 160))
countImages = 0

for images in range(0, len(imagesArray)):
    imageIndex = images
    imageDirectory = "G:/VNIR/"
    imageName = imagesArray[imageIndex]
    countImages = countImages + 1

    # instantiating hyperspectral image object using spectral library
    VNIRimage = np.load(imageDirectory + imageName + ".npy")

    VNIRnumOfRows = VNIRimage.shape[0]
    VNIRnumOfCols = VNIRimage.shape[1]
    VNIRnumOfBands = VNIRimage.shape[2]

    imgSpectrum = int(str(VNIRnumOfBands))
    imgHeight = int(str(VNIRnumOfRows))
    imgWidth = int(str(VNIRnumOfCols))

    imageNameShort = ""
    for imgIndex in range(0, 6):
        imageNameShort = imageNameShort + imageName[imgIndex]
    imageNameShort = int(imageNameShort)
    shortenImgNameArray[images] = imageNameShort

    for b in range(0, VNIRnumOfBands):
        intensityAve = np.average(VNIRimage[280:285, 10:310, b])
        intensityDataSet[images, b] = intensityAve

    print(str(images + 1) + " images of " + str(len(imagesArray)))

# //////////////////////////// export images intensity values to excel file and numpy array //////////////////////

intensityDataSet = np.concatenate((shortenImgNameArray, intensityDataSet), axis=1)

# np.save("F:/SWIR/dataset/" + "SWIR_dataset_based_on_mean_stddev", imgNameIntensityValueArray, allow_pickle=True, fix_imports=True)
np.savetxt("G:/dataset/" + "dayLightIntensityVNIR.csv", intensityDataSet, delimiter=',')
