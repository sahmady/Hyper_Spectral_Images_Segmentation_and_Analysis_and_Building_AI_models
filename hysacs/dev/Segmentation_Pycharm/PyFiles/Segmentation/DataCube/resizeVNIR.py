import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi


SWIRFileSource = "../../SegmentedImages/SourceFolder/VNIR.txt"
imageType = "VNIR"
imagesArray = []
with open(SWIRFileSource, "r") as listOfImages:
    for line in listOfImages:
        imagesArray.append(line.strip())

for images in range(421, len(imagesArray)):
    # image 150; 038425_20000_us_2x_2014-11-14T151756_corr_rad is out of bounds so skipped
    # image 242; 038622_20000_us_2x_2014-11-19T113700_corr_rad is out of bounds so skipped
    # image 338; 038723_20000_us_2x_2014-11-20T140454_corr_rad is out of bounds so skipped
    # image 421; 038810_20000_us_2x_2014-11-21T114620_corr_rad is out of bounds so skipped
    imageIndex = images
    imageDirectory = "D:/VNIR/"
    imageName = imagesArray[imageIndex]

    # instantiating hyperspectral image object using spectral library
    img = envi.open(imageDirectory + imageName + '.hdr',
                    imageDirectory + imageName + '.img')

    imgRows = img.nrows
    imgColumns = img.ncols
    imgBands = img.nbands

    imgSpectrum = int(str(imgBands))
    imgHeight = int(str(imgRows))
    imgWidth = int(str(imgColumns))
    newHeight = int(imgHeight / 5)
    newWidth = int(imgWidth / 5)

    image = np.zeros((newHeight, newWidth, imgBands))
    # /////////////////////////////////////////////////// resize based on mean ///////////////////////////////////////
    # for b in range(0, imgBands):
    #     for r in range(0, imgRows, 5):
    #         if r < 980:
    #             for c in range(0, imgColumns, 5):
    #                 mean = 0
    #                 for x in range(0, 5):
    #                     for y in range(0, 5):
    #                         mean = img[r + x, c + y, b] + mean
    #                 image[int(r / 5), int(c / 5), b] = mean / 25

    # ////////////////////////////////////////////////// resize based on median //////////////////////////////////////

    for b in range(0, imgBands):
        for r in range(0, imgRows, 5):
            if r < 1500:
                for c in range(0, imgColumns, 5):
                    # mean = 0
                    # for x in range(0, 5):
                    #     for y in range(0, 5):
                    #         mean = img[r + x, c + y, b] + mean
                    image[int(r / 5), int(c / 5), b] = img[r - 2, c - 2, b]


    # np.savetxt("../../SegmentedImages/" + imageType + "/" + imageName + '.csv', image[:, :, 0], delimiter=',')
    np.save("G:/VNIR/" + imageName, image, allow_pickle=True, fix_imports=True)

    print(str(images + 1) + " images of " + str(len(imagesArray)))

