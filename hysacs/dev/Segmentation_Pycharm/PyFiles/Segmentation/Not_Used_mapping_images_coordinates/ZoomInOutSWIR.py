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

for images in range(0, len(imagesArray)):
    imageIndex = images
    imageDirectory = "G:/VNIR/"
    imageName = imagesArray[imageIndex]

    VNIRimage = np.load(imageDirectory + imageName + ".npy")

    # print(image.shape)

    numOfRows = VNIRimage.shape[0]
    numOfCols = VNIRimage.shape[1]
    numOfBands = VNIRimage.shape[2]

    VNIRDamageSignChannels = np.array([23, 50, 100])
    VNIRDamageSignMax1 = np.max(VNIRimage[:, :, int(VNIRDamageSignChannels[0])])
    VNIRDamageSignMin1 = np.min(VNIRimage[:, :, int(VNIRDamageSignChannels[0])])
    VNIRDamageSignMax2 = np.max(VNIRimage[:, :, int(VNIRDamageSignChannels[1])])
    VNIRDamageSignMin2 = np.min(VNIRimage[:, :, int(VNIRDamageSignChannels[1])])
    VNIRDamageSignMax3 = np.max(VNIRimage[:, :, int(VNIRDamageSignChannels[2])])
    VNIRDamageSignMin3 = np.min(VNIRimage[:, :, int(VNIRDamageSignChannels[2])])

    VNIRDamageSignslice1 = np.zeros((int(numOfRows), int(numOfCols)))
    VNIRDamageSignslice2 = np.zeros((int(numOfRows), int(numOfCols)))
    VNIRDamageSignslice3 = np.zeros((int(numOfRows), int(numOfCols)))

    for r in range(0, numOfRows):
        if r > 210:
            for c in range(0, int(numOfCols)):
                VNIRDamageSignslice1[r, c] = (VNIRimage[r, c, int(VNIRDamageSignChannels[0])] - VNIRDamageSignMin1) / \
                                             (VNIRDamageSignMax1 - VNIRDamageSignMin1) * 255
                VNIRDamageSignslice2[r, c] = (VNIRimage[r, c, int(VNIRDamageSignChannels[1])] - VNIRDamageSignMin3) / \
                                             (VNIRDamageSignMax3 - VNIRDamageSignMin3) * 255
                VNIRDamageSignslice3[r, c] = (VNIRimage[r, c, int(VNIRDamageSignChannels[2])] - VNIRDamageSignMin2) / \
                                             (VNIRDamageSignMax2 - VNIRDamageSignMin2) * 255

    VNIRDamageSignrgbImage = np.zeros([int(numOfRows), numOfCols, 3], dtype=np.uint8)
    for r in range(0, int(numOfRows)):
        for c in range(0, int(numOfCols)):
            VNIRDamageSignrgbImage[r, c] = [VNIRDamageSignslice1[r, c],
                                            VNIRDamageSignslice2[r, c],
                                            VNIRDamageSignslice3[r, c]]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRDamageSignrgbImage)

    # ///////////////////////////////////////////// segment VNIRY1Y2Y3X2 points //////////////////////////////////
    # nemo = cv2.imread(imageDirectory + imageName)
    VNIRDamageSignsegmentrgb = VNIRDamageSignrgbImage
    # plt.imshow(segmentrgb)
    # plt.show()
    VNIRDamageSignsegmented3d = cv2.cvtColor(VNIRDamageSignsegmentrgb, cv2.COLOR_RGB2BGR)
    # plt.imshow(segmented3d)
    # plt.show()
    DamageSignhsvconverted = cv2.cvtColor(VNIRDamageSignsegmented3d, cv2.COLOR_RGB2HSV)
    DamageSign_lower_bounds = (150, 0, 0)
    DamageSign_upper_bounds = (177, 255, 255)
    DamageSignmask = cv2.inRange(DamageSignhsvconverted, DamageSign_lower_bounds, DamageSign_upper_bounds)
    DamageSignresult = cv2.bitwise_and(VNIRDamageSignsegmented3d, VNIRDamageSignsegmented3d, mask=DamageSignmask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    DamageSignresult = cv2.cvtColor(DamageSignresult, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignresult)

    DamageSignsegmented = np.zeros((numOfRows, numOfCols))
    for r in range(0, numOfRows):
        for c in range(0, numOfCols):
            if DamageSignresult[r, c, 0] or DamageSignresult[r, c, 1] or DamageSignresult[r, c, 2]:
                DamageSignsegmented[r, c] = DamageSignresult[r, c, 2]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignsegmented)

    # removing lines zero degrees
    noOfPixels = 10
    for r in range(0, numOfRows):
        for c in range(0, numOfCols - noOfPixels):
            if DamageSignsegmented[r, c] > 0:
                countPixels = 0
                for i in range(0, noOfPixels):
                    if DamageSignsegmented[r, c + i] > 0:
                        countPixels = countPixels + 1
                if countPixels == noOfPixels:
                    for d in range(0, noOfPixels):
                        DamageSignsegmented[r, c + d] = 0

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignsegmented)

    # ///////////////////////////////////////////////// structuring elements //////////////////////////////////
    """
    structuring elements for noise removal
    radius is based on Moore neighborhood
    """
    def remove_noise_points(radius, minPoints):
        labels = np.zeros((numOfRows, numOfCols))

        for r in range(0, numOfRows):
            for c in range(0, numOfCols):
                if DamageSignsegmented[r, c] > 0:
                    labels[r, c] = 1

        for r in range(radius - 1, numOfRows - radius):
            for c in range(radius - 1, numOfCols - radius):
                if DamageSignsegmented[r, c] > 0:
                    numberOfNieghbors = 0
                    for radRow in range(r - radius + 1, r + radius):
                        for radCol in range(c - radius + 1, c + radius):
                            numberOfNieghbors = numberOfNieghbors + labels[radRow, radCol]
                    if numberOfNieghbors < minPoints:
                        DamageSignsegmented[r, c] = 0


    # cleaning grains
    radius = 4
    minPoints = 15
    iterations = 2
    for i in range(0, iterations):
        remove_noise_points(radius, minPoints)

    # radius = 4
    # minPoints = 32
    # iterations = 3
    # for i in range(0, iterations):
    #     remove_noise_points(radius, minPoints)

    cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignsegmented)

    # //////////////////////////////////////////// find VNIRY1 and VNIRY2 point //////////////////////////////

    DamageSignXdistance = 0
    for r in range(210, 350):
        count = 0
        for c in range(0, numOfCols):
            if DamageSignsegmented[r, c] > 0:
                DamageSignXdistance = count
                break
            count = count + 1

    print(DamageSignXdistance) # two points ahead (DamageSignXdistance-2 will give the mid point)




    #
    #
    #
    # # # zero padding
    # # threshold = 100
    # # for r in range(0, numOfRows):
    # #     for c in range(0, numOfCols):
    # #         if slice1[r, c] < threshold:
    # #             slice1[r, c] = 0
    # #         elif slice1[r, c] > 220:
    # #             slice1[r, c] = 255
    # #
    # # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", slice1)
    #
    # # //////////////////////////////////////////// export data as image or excel file //////////////////////////////
    # # cv2.imwrite("../../SegmentedImages/" + "/" + "1.png", image[:, :, 10])
    # # segSlice1 = np.zeros((imgRows, imgColumns))
    print(str(images + 1) + " images of " + str(len(imagesArray)))
