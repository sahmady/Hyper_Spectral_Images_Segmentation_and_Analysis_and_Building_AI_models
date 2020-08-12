import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi

SWIRFileSource = "../../SegmentedImages/SourceFolder/SWIR.txt"
SWIRimageType = "SWIR"
SWIRimagesArray = []
with open(SWIRFileSource, "r") as SWIRlistOfImages:
    for line in SWIRlistOfImages:
        SWIRimagesArray.append(line.strip())

for SWIRimages in range(0, len(SWIRimagesArray)):
    SWIRimageIndex = SWIRimages
    SWIRimageDirectory = "F:/SWIR/"
    SWIRimageName = SWIRimagesArray[SWIRimageIndex]

    SWIRimage = envi.open(SWIRimageDirectory + SWIRimageName + '.hdr',
                          SWIRimageDirectory + SWIRimageName + '.img')

    SWIRnumOfRows = SWIRimage.nrows
    SWIRnumOfCols = SWIRimage.ncols
    SWIRnumOfBands = SWIRimage.nbands

    # ///////////////////////////////////// SWIR coordinating points /////////////////////////////////////////
    SWIRX1channels = np.array([230, 80, 120])
    SWIRX1channelsMax1 = np.max(SWIRimage[:, :, int(SWIRX1channels[0])])
    SWIRX1channelsMin1 = np.min(SWIRimage[:, :, int(SWIRX1channels[0])])
    SWIRX1channelsMax2 = np.max(SWIRimage[:, :, int(SWIRX1channels[1])])
    SWIRX1channelsMin2 = np.min(SWIRimage[:, :, int(SWIRX1channels[1])])
    SWIRX1channelsMax3 = np.max(SWIRimage[:, :, int(SWIRX1channels[2])])
    SWIRX1channelsMin3 = np.min(SWIRimage[:, :, int(SWIRX1channels[2])])

    SWIRY1Y2Y3X2channels = np.array([230, 240, 250])
    SWIRY1Y2Y3X2Max1 = np.max(SWIRimage[:, :, int(SWIRY1Y2Y3X2channels[0])])
    SWIRY1Y2Y3X2Min1 = np.min(SWIRimage[:, :, int(SWIRY1Y2Y3X2channels[0])])
    SWIRY1Y2Y3X2Max2 = np.max(SWIRimage[:, :, int(SWIRY1Y2Y3X2channels[1])])
    SWIRY1Y2Y3X2Min2 = np.min(SWIRimage[:, :, int(SWIRY1Y2Y3X2channels[1])])
    SWIRY1Y2Y3X2Max3 = np.max(SWIRimage[:, :, int(SWIRY1Y2Y3X2channels[2])])
    SWIRY1Y2Y3X2Min3 = np.min(SWIRimage[:, :, int(SWIRY1Y2Y3X2channels[2])])

    # SWIRDamageSignChannels = np.array([23, 50, 100])
    # SWIRDamageSignMax1 = np.max(SWIRimage[:, :, int(SWIRDamageSignChannels[0])])
    # SWIRDamageSignMin1 = np.min(SWIRimage[:, :, int(SWIRDamageSignChannels[0])])
    # SWIRDamageSignMax2 = np.max(SWIRimage[:, :, int(SWIRDamageSignChannels[1])])
    # SWIRDamageSignMin2 = np.min(SWIRimage[:, :, int(SWIRDamageSignChannels[1])])
    # SWIRDamageSignMax3 = np.max(SWIRimage[:, :, int(SWIRDamageSignChannels[2])])
    # SWIRDamageSignMin3 = np.min(SWIRimage[:, :, int(SWIRDamageSignChannels[2])])

    SWIRX1channelsslice1 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    SWIRX1channelsslice2 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    SWIRX1channelsslice3 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))

    SWIRY1Y2Y3X2slice1 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    SWIRY1Y2Y3X2slice2 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    SWIRY1Y2Y3X2slice3 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))

    # SWIRDamageSignslice1 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    # SWIRDamageSignslice2 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    # SWIRDamageSignslice3 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))

    for r in range(0, SWIRnumOfRows):
        if 450 < r < 610:
            for c in range(0, 100):
                SWIRX1channelsslice1[r, c] = (SWIRimage[r, c, int(SWIRX1channels[0])] - SWIRX1channelsMin1) / \
                                             (SWIRX1channelsMax1 - SWIRX1channelsMin1) * 255
                SWIRX1channelsslice2[r, c] = (SWIRimage[r, c, int(SWIRX1channels[1])] - SWIRX1channelsMin3) / \
                                             (SWIRX1channelsMax3 - SWIRX1channelsMin3) * 255
                SWIRX1channelsslice3[r, c] = (SWIRimage[r, c, int(SWIRX1channels[2])] - SWIRX1channelsMin2) / \
                                             (SWIRX1channelsMax2 - SWIRX1channelsMin2) * 255

    for r in range(0, SWIRnumOfRows):
        if r > 660:
            for c in range(0, int(SWIRnumOfCols)):
                SWIRY1Y2Y3X2slice1[r, c] = (SWIRimage[r, c, int(SWIRY1Y2Y3X2channels[0])] - SWIRY1Y2Y3X2Min1) / \
                                           (SWIRY1Y2Y3X2Max1 - SWIRY1Y2Y3X2Min1) * 255
                SWIRY1Y2Y3X2slice2[r, c] = (SWIRimage[r, c, int(SWIRY1Y2Y3X2channels[1])] - SWIRY1Y2Y3X2Min3) / \
                                           (SWIRY1Y2Y3X2Max3 - SWIRY1Y2Y3X2Min3) * 255
                SWIRY1Y2Y3X2slice3[r, c] = (SWIRimage[r, c, int(SWIRY1Y2Y3X2channels[2])] - SWIRY1Y2Y3X2Min2) / \
                                           (SWIRY1Y2Y3X2Max2 - SWIRY1Y2Y3X2Min2) * 255

    # for r in range(0, SWIRnumOfRows):
    #     if r > 210:
    #         for c in range(0, int(SWIRnumOfCols)):
    #             SWIRDamageSignslice1[r, c] = (SWIRimage[r, c, int(SWIRDamageSignChannels[0])] - SWIRDamageSignMin1) / \
    #                                          (SWIRDamageSignMax1 - SWIRDamageSignMin1) * 255
    #             SWIRDamageSignslice2[r, c] = (SWIRimage[r, c, int(SWIRDamageSignChannels[1])] - SWIRDamageSignMin3) / \
    #                                          (SWIRDamageSignMax3 - SWIRDamageSignMin3) * 255
    #             SWIRDamageSignslice3[r, c] = (SWIRimage[r, c, int(SWIRDamageSignChannels[2])] - SWIRDamageSignMin2) / \
    #                                          (SWIRDamageSignMax2 - SWIRDamageSignMin2) * 255

    SWIRX1channelsrgbImage = np.zeros([int(SWIRnumOfRows), SWIRnumOfCols, 3], dtype=np.uint8)
    for r in range(0, int(SWIRnumOfRows)):
        if 450 < r < 610:
            for c in range(0, 100):
                SWIRX1channelsrgbImage[r, c] = [SWIRX1channelsslice1[r, c],
                                                SWIRX1channelsslice2[r, c],
                                                SWIRX1channelsslice3[r, c]]

    SWIRY1Y2Y3X2rgbImage = np.zeros([int(SWIRnumOfRows), SWIRnumOfCols, 3], dtype=np.uint8)
    for r in range(0, int(SWIRnumOfRows)):
        if r > 660:
            for c in range(0, int(SWIRnumOfCols)):
                SWIRY1Y2Y3X2rgbImage[r, c] = [SWIRY1Y2Y3X2slice1[r, c],
                                              SWIRY1Y2Y3X2slice2[r, c],
                                              SWIRY1Y2Y3X2slice3[r, c]]

    # SWIRDamageSignrgbImage = np.zeros([int(SWIRnumOfRows), SWIRnumOfCols, 3], dtype=np.uint8)
    # for r in range(0, int(SWIRnumOfRows)):
    #     for c in range(0, int(SWIRnumOfCols)):
    #         SWIRDamageSignrgbImage[r, c] = [SWIRDamageSignslice1[r, c],
    #                                         SWIRDamageSignslice2[r, c],
    #                                         SWIRDamageSignslice3[r, c]]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRX1channelsrgbImage)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRY1Y2Y3X2rgbImage)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRDamageSignrgbImage)

    # ///////////////////////////////////////// segment SWIRX1 and SWIRY1Y2Y3X2 point ////////////////////////////
    # nemo = cv2.imread(imageDirectory + imageName)
    SWIRX1segmentrgb = SWIRX1channelsrgbImage
    # plt.imshow(segmentrgb)
    # plt.show()
    SWIRX1segmented3d = cv2.cvtColor(SWIRX1segmentrgb, cv2.COLOR_RGB2BGR)
    # plt.imshow(segmented3d)
    # plt.show()
    SWIRX1hsvconverted = cv2.cvtColor(SWIRX1segmented3d, cv2.COLOR_RGB2HSV)
    SWIRX1_lower_bounds = (40, 0, 0)  # light_range = (0, 60, 50) works perfectly for 103, 47, 31 channels
    SWIRX1_upper_bounds = (50, 255, 255)
    SWIRX1mask = cv2.inRange(SWIRX1hsvconverted, SWIRX1_lower_bounds, SWIRX1_upper_bounds)
    SWIRX1result = cv2.bitwise_and(SWIRX1segmented3d, SWIRX1segmented3d, mask=SWIRX1mask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    SWIRX1result = cv2.cvtColor(SWIRX1result, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRX1result)

    # nemo = cv2.imread(imageDirectory + imageName)
    SWIRY1Y2Y3X2segmentrgb = SWIRY1Y2Y3X2rgbImage
    # plt.imshow(segmentrgb)
    # plt.show()
    SWIRY1Y2Y3X2segmented3d = cv2.cvtColor(SWIRY1Y2Y3X2segmentrgb, cv2.COLOR_RGB2BGR)
    # plt.imshow(segmented3d)
    # plt.show()
    SWIRY1Y2Y3X2hsvconverted = cv2.cvtColor(SWIRY1Y2Y3X2segmented3d, cv2.COLOR_RGB2HSV)
    SWIRY1Y2Y3X2_lower_bounds = (0, 0, 100)  # light_range = (0, 60, 50) works perfectly for 103, 47, 31 channels
    SWIRY1Y2Y3X2_upper_bounds = (100, 255, 255)
    SWIRY1Y2Y3X2mask = cv2.inRange(SWIRY1Y2Y3X2hsvconverted, SWIRY1Y2Y3X2_lower_bounds, SWIRY1Y2Y3X2_upper_bounds)
    SWIRY1Y2Y3X2result = cv2.bitwise_and(SWIRY1Y2Y3X2segmented3d, SWIRY1Y2Y3X2segmented3d, mask=SWIRY1Y2Y3X2mask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    SWIRY1Y2Y3X2result = cv2.cvtColor(SWIRY1Y2Y3X2result, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRY1Y2Y3X2result)

    # # nemo = cv2.imread(imageDirectory + imageName)
    # SWIRDamageSignsegmentrgb = SWIRDamageSignrgbImage
    # # plt.imshow(segmentrgb)
    # # plt.show()
    # SWIRDamageSignsegmented3d = cv2.cvtColor(SWIRDamageSignsegmentrgb, cv2.COLOR_RGB2BGR)
    # # plt.imshow(segmented3d)
    # # plt.show()
    # SWIRDamageSignhsvconverted = cv2.cvtColor(SWIRDamageSignsegmented3d, cv2.COLOR_RGB2HSV)
    # SWIRDamageSign_lower_bounds = (150, 0, 0)
    # SWIRDamageSign_upper_bounds = (177, 255, 255)
    # SWIRDamageSignmask = cv2.inRange(SWIRDamageSignhsvconverted,
    #                                  SWIRDamageSign_lower_bounds,
    #                                  SWIRDamageSign_upper_bounds)
    # SWIRDamageSignresult = cv2.bitwise_and(SWIRDamageSignsegmented3d,
    #                                        SWIRDamageSignsegmented3d,
    #                                        mask=SWIRDamageSignmask)
    # # plt.subplot(1, 2, 1)
    # # plt.imshow(mask, cmap="gray")
    # # plt.subplot(1, 2, 2)
    # # plt.imshow(result)
    # # plt.show()
    # SWIRDamageSignresult = cv2.cvtColor(SWIRDamageSignresult, cv2.COLOR_BGR2RGB)
    # # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignresult)

    # /////////////////////////////////////////////// keep only one slice /////////////////////////////////////////

    SWIRX1segmented = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    for r in range(0, SWIRnumOfRows):
        for c in range(0, SWIRnumOfCols):
            if SWIRX1result[r, c, 0] or SWIRX1result[r, c, 1] or SWIRX1result[r, c, 2]:
                SWIRX1segmented[r, c] = SWIRX1result[r, c, 2]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRX1segmented)

    SWIRY1Y2Y3X2segmented = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    for r in range(0, SWIRnumOfRows):
        for c in range(0, SWIRnumOfCols):
            if SWIRY1Y2Y3X2result[r, c, 0] or SWIRY1Y2Y3X2result[r, c, 1] or SWIRY1Y2Y3X2result[r, c, 2]:
                SWIRY1Y2Y3X2segmented[r, c] = SWIRY1Y2Y3X2result[r, c, 2]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRY1Y2Y3X2segmented)

    # SWIRDamageSignsegmented = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    # for r in range(0, SWIRnumOfRows):
    #     for c in range(0, SWIRnumOfCols):
    #         if SWIRDamageSignresult[r, c, 0] or SWIRDamageSignresult[r, c, 1] or SWIRDamageSignresult[r, c, 2]:
    #             SWIRDamageSignsegmented[r, c] = SWIRDamageSignresult[r, c, 2]
    #
    # # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignsegmented)

    # //////////////////////////////////////////// removing lines zero degrees ////////////////////////////////
    # noOfPixels = 10
    # for r in range(0, SWIRnumOfRows):
    #     for c in range(0, SWIRnumOfCols - noOfPixels):
    #         if SWIRDamageSignsegmented[r, c] > 0:
    #             countPixels = 0
    #             for i in range(0, noOfPixels):
    #                 if SWIRDamageSignsegmented[r, c + i] > 0:
    #                     countPixels = countPixels + 1
    #             if countPixels == noOfPixels:
    #                 for d in range(0, noOfPixels):
    #                     SWIRDamageSignsegmented[r, c + d] = 0

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignsegmented)

    """
    structuring elements for noise removal radius is based on Moore neighborhood
    """

    def remove_noise_points(radius, minPoints, image):
        labels = np.zeros((SWIRnumOfRows, SWIRnumOfCols))

        segmented = image

        for r in range(0, SWIRnumOfRows):
            for c in range(0, SWIRnumOfCols):
                if segmented[r, c] > 0:
                    labels[r, c] = 1

        for r in range(radius - 1, SWIRnumOfRows - radius):
            for c in range(radius - 1, SWIRnumOfCols - radius):
                if segmented[r, c] > 0:
                    numberOfNieghbors = 0
                    for radRow in range(r - radius + 1, r + radius):
                        for radCol in range(c - radius + 1, c + radius):
                            numberOfNieghbors = numberOfNieghbors + labels[radRow, radCol]
                    if numberOfNieghbors < minPoints:
                        segmented[r, c] = 0


    radius = 5
    minPoints = 25
    iterations = 2
    for i in range(0, iterations):
        remove_noise_points(radius, minPoints, SWIRX1segmented)

    radius = 3
    iterations = 1
    minPoints = 10
    for i in range(0, iterations):
        remove_noise_points(radius, minPoints, SWIRY1Y2Y3X2segmented)

    # radius = 4
    # minPoints = 15
    # iterations = 2
    # for i in range(0, iterations):
    #     remove_noise_points(radius, minPoints, SWIRDamageSignsegmented)

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRX1segmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRY1Y2Y3X2segmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRDamageSignsegmented)

    # //////////////////////////////// find SWIRX1 and SWIRY1Y2Y3X2 and damagesign points //////////////////////////

    SWIRX1distance = 0
    for c in range(100, 0, -1):
        count = 0
        SWIRX1distance = 0
        for r in range(0, SWIRnumOfRows):
            if SWIRX1segmented[r, c] > 0:
                count = count + 1
        if count >= 4:
            SWIRX1distance = c
            break

    c = 20
    count = 0
    for r in range(0, SWIRnumOfRows):
        if SWIRY1Y2Y3X2segmented[r, c] > 0:
            break
        count = count + 1
    SWIRY1distance = count

    c = SWIRnumOfCols - 20
    count = 0
    for r in range(0, SWIRnumOfRows):
        if SWIRY1Y2Y3X2segmented[r, c] > 0:
            break
        count = count + 1
    SWIRY2distance = count

    # DamageSignXdistance = 0
    # for r in range(210, 350):
    #     count = 0
    #     for c in range(0, SWIRnumOfCols):
    #         if SWIRDamageSignsegmented[r, c] > 0:
    #             DamageSignXdistance = count
    #             break
    #         count = count + 1


    print(SWIRX1distance)
    print(SWIRY1distance)
    print(SWIRY2distance)
    # print(DamageSignXdistance)

    # //////////////////////////////////////////// export data as image or excel file //////////////////////////////
    # cv2.imwrite("../../SegmentedImages/" + "/" + "1.png", image[:, :, 10])
    # segSlice1 = np.zeros((imgRows, imgColumns))
    print(str(SWIRimages + 1) + " images of " + str(len(SWIRimagesArray)))

