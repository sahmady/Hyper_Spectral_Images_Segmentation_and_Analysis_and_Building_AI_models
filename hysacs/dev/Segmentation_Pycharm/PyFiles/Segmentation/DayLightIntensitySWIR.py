import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import pandas as pd

SWIRFileSource = "../../SegmentedImages/SourceFolder/SWIR.txt"
imageType = "SWIR"
imagesArray = []
with open(SWIRFileSource, "r") as listOfImages:
    for line in listOfImages:
        imagesArray.append(line.strip())

shortenImgNameArray = np.zeros((len(imagesArray), 1))
intensityDataSet = np.zeros((len(imagesArray), 256))
countImages = 0

for images in range(0, len(imagesArray)):
    imageIndex = images
    imageDirectory = "F:/SWIR/"
    imageName = imagesArray[imageIndex]
    countImages = countImages + 1

    # instantiating hyperspectral image object using spectral library
    SWIRimage = envi.open(imageDirectory + imageName + '.hdr',
                          imageDirectory + imageName + '.img')

    SWIRnumOfRows = SWIRimage.nrows
    SWIRnumOfCols = SWIRimage.ncols
    SWIRnumOfBands = SWIRimage.nbands

    imgSpectrum = int(str(SWIRnumOfBands))
    imgHeight = int(str(SWIRnumOfRows))
    imgWidth = int(str(SWIRnumOfCols))

    imageNameShort = ""
    for imgIndex in range(0, 6):
        imageNameShort = imageNameShort + imageName[imgIndex]
    imageNameShort = int(imageNameShort)
    shortenImgNameArray[images] = imageNameShort

    for b in range(0, SWIRnumOfBands):
        intensityAve = np.average(SWIRimage[780:785, 10:310, b])
        intensityDataSet[images, b] = intensityAve

    print(str(images + 1) + " images of " + str(len(imagesArray)))

# //////////////////////////// export images intensity values to excel file and numpy array //////////////////////

intensityDataSet = np.concatenate((shortenImgNameArray, intensityDataSet), axis=1)

# np.save("F:/SWIR/dataset/" + "SWIR_dataset_based_on_mean_stddev", imgNameIntensityValueArray, allow_pickle=True, fix_imports=True)
np.savetxt("F:/SWIR/dataset/" + "dayLightIntensitySWIR.csv", intensityDataSet, delimiter=',')
