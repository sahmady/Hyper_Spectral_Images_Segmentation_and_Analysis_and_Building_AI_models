import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi

VNIRFileSource = "../../SegmentedImages/SourceFolder/VNIR.txt"
VNIRimageType = "VNIR"
VNIRimagesArray = []
with open(VNIRFileSource, "r") as VNIRlistOfImages:
    for line in VNIRlistOfImages:
        VNIRimagesArray.append(line.strip())

for VNIRimages in range(0, len(VNIRimagesArray)):
    VNIRimageIndex = VNIRimages
    VNIRimageDirectory = "G:/VNIR/"
    VNIRimageName = VNIRimagesArray[VNIRimageIndex]

    VNIRimage = np.load(VNIRimageDirectory + VNIRimageName + ".npy")

    VNIRnumOfRows = VNIRimage.shape[0]
    VNIRnumOfCols = VNIRimage.shape[1]
    VNIRnumOfBands = VNIRimage.shape[2]

    # ///////////////////////////////////// VNIR coordinating points /////////////////////////////////////////
    VNIRX1channels = np.array([23, 31, 39])
    VNIRX1channelsMax1 = np.max(VNIRimage[:, :, int(VNIRX1channels[0])])
    VNIRX1channelsMin1 = np.min(VNIRimage[:, :, int(VNIRX1channels[0])])
    VNIRX1channelsMax2 = np.max(VNIRimage[:, :, int(VNIRX1channels[1])])
    VNIRX1channelsMin2 = np.min(VNIRimage[:, :, int(VNIRX1channels[1])])
    VNIRX1channelsMax3 = np.max(VNIRimage[:, :, int(VNIRX1channels[2])])
    VNIRX1channelsMin3 = np.min(VNIRimage[:, :, int(VNIRX1channels[2])])

    VNIRY1Y2Y3X2channels = np.array([119, 19, 19])
    VNIRY1Y2Y3X2Max1 = np.max(VNIRimage[:, :, int(VNIRY1Y2Y3X2channels[0])])
    VNIRY1Y2Y3X2Min1 = np.min(VNIRimage[:, :, int(VNIRY1Y2Y3X2channels[0])])
    VNIRY1Y2Y3X2Max2 = np.max(VNIRimage[:, :, int(VNIRY1Y2Y3X2channels[1])])
    VNIRY1Y2Y3X2Min2 = np.min(VNIRimage[:, :, int(VNIRY1Y2Y3X2channels[1])])
    VNIRY1Y2Y3X2Max3 = np.max(VNIRimage[:, :, int(VNIRY1Y2Y3X2channels[2])])
    VNIRY1Y2Y3X2Min3 = np.min(VNIRimage[:, :, int(VNIRY1Y2Y3X2channels[2])])

    VNIRDamageSignChannels = np.array([23, 50, 100])
    VNIRDamageSignMax1 = np.max(VNIRimage[:, :, int(VNIRDamageSignChannels[0])])
    VNIRDamageSignMin1 = np.min(VNIRimage[:, :, int(VNIRDamageSignChannels[0])])
    VNIRDamageSignMax2 = np.max(VNIRimage[:, :, int(VNIRDamageSignChannels[1])])
    VNIRDamageSignMin2 = np.min(VNIRimage[:, :, int(VNIRDamageSignChannels[1])])
    VNIRDamageSignMax3 = np.max(VNIRimage[:, :, int(VNIRDamageSignChannels[2])])
    VNIRDamageSignMin3 = np.min(VNIRimage[:, :, int(VNIRDamageSignChannels[2])])

    VNIRX1channelsslice1 = np.zeros((int(VNIRnumOfRows / 4), int(VNIRnumOfCols / 3)))
    VNIRX1channelsslice2 = np.zeros((int(VNIRnumOfRows / 4), int(VNIRnumOfCols / 3)))
    VNIRX1channelsslice3 = np.zeros((int(VNIRnumOfRows / 4), int(VNIRnumOfCols / 3)))

    VNIRY1Y2Y3X2slice1 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))
    VNIRY1Y2Y3X2slice2 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))
    VNIRY1Y2Y3X2slice3 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))

    VNIRDamageSignslice1 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))
    VNIRDamageSignslice2 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))
    VNIRDamageSignslice3 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))

    for r in range(0, VNIRnumOfRows):
        if 100 < r < 150:
            for c in range(0, int(VNIRnumOfCols / 3)):
                VNIRX1channelsslice1[r, c] = (VNIRimage[r, c, int(VNIRX1channels[0])] - VNIRX1channelsMin1) / \
                                             (VNIRX1channelsMax1 - VNIRX1channelsMin1) * 255
                VNIRX1channelsslice2[r, c] = (VNIRimage[r, c, int(VNIRX1channels[1])] - VNIRX1channelsMin3) / \
                                             (VNIRX1channelsMax3 - VNIRX1channelsMin3) * 255
                VNIRX1channelsslice3[r, c] = (VNIRimage[r, c, int(VNIRX1channels[2])] - VNIRX1channelsMin2) / \
                                             (VNIRX1channelsMax2 - VNIRX1channelsMin2) * 255

    for r in range(0, VNIRnumOfRows):
        if r > 210:
            for c in range(0, int(VNIRnumOfCols)):
                VNIRY1Y2Y3X2slice1[r, c] = (VNIRimage[r, c, int(VNIRY1Y2Y3X2channels[0])] - VNIRY1Y2Y3X2Min1) / \
                                           (VNIRY1Y2Y3X2Max1 - VNIRY1Y2Y3X2Min1) * 255
                VNIRY1Y2Y3X2slice2[r, c] = (VNIRimage[r, c, int(VNIRY1Y2Y3X2channels[1])] - VNIRY1Y2Y3X2Min3) / \
                                           (VNIRY1Y2Y3X2Max3 - VNIRY1Y2Y3X2Min3) * 255
                VNIRY1Y2Y3X2slice3[r, c] = (VNIRimage[r, c, int(VNIRY1Y2Y3X2channels[2])] - VNIRY1Y2Y3X2Min2) / \
                                           (VNIRY1Y2Y3X2Max2 - VNIRY1Y2Y3X2Min2) * 255

    for r in range(0, VNIRnumOfRows):
        if r > 210:
            for c in range(0, int(VNIRnumOfCols)):
                VNIRDamageSignslice1[r, c] = (VNIRimage[r, c, int(VNIRDamageSignChannels[0])] - VNIRDamageSignMin1) / \
                                             (VNIRDamageSignMax1 - VNIRDamageSignMin1) * 255
                VNIRDamageSignslice2[r, c] = (VNIRimage[r, c, int(VNIRDamageSignChannels[1])] - VNIRDamageSignMin3) / \
                                             (VNIRDamageSignMax3 - VNIRDamageSignMin3) * 255
                VNIRDamageSignslice3[r, c] = (VNIRimage[r, c, int(VNIRDamageSignChannels[2])] - VNIRDamageSignMin2) / \
                                             (VNIRDamageSignMax2 - VNIRDamageSignMin2) * 255

    VNIRX1channelsrgbImage = np.zeros([int(VNIRnumOfRows), VNIRnumOfCols, 3], dtype=np.uint8)
    for r in range(0, int(VNIRnumOfRows / 4)):
        for c in range(0, int(VNIRnumOfCols / 3)):
            VNIRX1channelsrgbImage[r, c] = [VNIRX1channelsslice1[r, c],
                                            VNIRX1channelsslice2[r, c],
                                            VNIRX1channelsslice3[r, c]]

    VNIRY1Y2Y3X2rgbImage = np.zeros([int(VNIRnumOfRows), VNIRnumOfCols, 3], dtype=np.uint8)
    for r in range(0, int(VNIRnumOfRows)):
        for c in range(0, int(VNIRnumOfCols)):
            VNIRY1Y2Y3X2rgbImage[r, c] = [VNIRY1Y2Y3X2slice1[r, c],
                                          VNIRY1Y2Y3X2slice2[r, c],
                                          VNIRY1Y2Y3X2slice3[r, c]]

    VNIRDamageSignrgbImage = np.zeros([int(VNIRnumOfRows), VNIRnumOfCols, 3], dtype=np.uint8)
    for r in range(0, int(VNIRnumOfRows)):
        for c in range(0, int(VNIRnumOfCols)):
            VNIRDamageSignrgbImage[r, c] = [VNIRDamageSignslice1[r, c],
                                            VNIRDamageSignslice2[r, c],
                                            VNIRDamageSignslice3[r, c]]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRX1channelsrgbImage)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRY1Y2Y3X2rgbImage)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRDamageSignrgbImage)

    # ///////////////////////////////////////// segment VNIRX1 and VNIRY1Y2Y3X2 point ////////////////////////////
    # nemo = cv2.imread(imageDirectory + imageName)
    VNIRX1segmentrgb = VNIRX1channelsrgbImage
    # plt.imshow(segmentrgb)
    # plt.show()
    VNIRX1segmented3d = cv2.cvtColor(VNIRX1segmentrgb, cv2.COLOR_RGB2BGR)
    # plt.imshow(segmented3d)
    # plt.show()
    VNIRX1hsvconverted = cv2.cvtColor(VNIRX1segmented3d, cv2.COLOR_RGB2HSV)
    VNIRX1_lower_bounds = (10, 0, 80)  # light_range = (0, 60, 50) works perfectly for 103, 47, 31 channels
    VNIRX1_upper_bounds = (20, 255, 255)
    VNIRX1mask = cv2.inRange(VNIRX1hsvconverted, VNIRX1_lower_bounds, VNIRX1_upper_bounds)
    VNIRX1result = cv2.bitwise_and(VNIRX1segmented3d, VNIRX1segmented3d, mask=VNIRX1mask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    VNIRX1result = cv2.cvtColor(VNIRX1result, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRX1result)

    # nemo = cv2.imread(imageDirectory + imageName)
    VNIRY1Y2Y3X2segmentrgb = VNIRY1Y2Y3X2rgbImage
    # plt.imshow(segmentrgb)
    # plt.show()
    VNIRY1Y2Y3X2segmented3d = cv2.cvtColor(VNIRY1Y2Y3X2segmentrgb, cv2.COLOR_RGB2BGR)
    # plt.imshow(segmented3d)
    # plt.show()
    Y1Y2Y3X2hsvconverted = cv2.cvtColor(VNIRY1Y2Y3X2segmented3d, cv2.COLOR_RGB2HSV)
    Y1Y2Y3X2_lower_bounds = (20, 0, 0)  # light_range = (0, 60, 50) works perfectly for 103, 47, 31 channels
    Y1Y2Y3X2_upper_bounds = (50, 255, 255)
    Y1Y2Y3X2mask = cv2.inRange(Y1Y2Y3X2hsvconverted, Y1Y2Y3X2_lower_bounds, Y1Y2Y3X2_upper_bounds)
    Y1Y2Y3X2result = cv2.bitwise_and(VNIRY1Y2Y3X2segmented3d, VNIRY1Y2Y3X2segmented3d, mask=Y1Y2Y3X2mask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    Y1Y2Y3X2result = cv2.cvtColor(Y1Y2Y3X2result, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", Y1Y2Y3X2result)

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

    # /////////////////////////////////////////////// keep only one slice /////////////////////////////////////////

    VNIRX1segmented = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    for r in range(0, VNIRnumOfRows):
        for c in range(0, VNIRnumOfCols):
            if VNIRX1result[r, c, 0] or VNIRX1result[r, c, 1] or VNIRX1result[r, c, 2]:
                VNIRX1segmented[r, c] = VNIRX1result[r, c, 2]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRX1segmented)

    Y1Y2Y3X2segmented = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    for r in range(0, VNIRnumOfRows):
        for c in range(0, VNIRnumOfCols):
            if Y1Y2Y3X2result[r, c, 0] or Y1Y2Y3X2result[r, c, 1] or Y1Y2Y3X2result[r, c, 2]:
                Y1Y2Y3X2segmented[r, c] = Y1Y2Y3X2result[r, c, 2]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", Y1Y2Y3X2segmented)

    DamageSignsegmented = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    for r in range(0, VNIRnumOfRows):
        for c in range(0, VNIRnumOfCols):
            if DamageSignresult[r, c, 0] or DamageSignresult[r, c, 1] or DamageSignresult[r, c, 2]:
                DamageSignsegmented[r, c] = DamageSignresult[r, c, 2]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignsegmented)

    # //////////////////////////////////////////// removing lines zero degrees ////////////////////////////////
    noOfPixels = 10
    for r in range(0, VNIRnumOfRows):
        for c in range(0, VNIRnumOfCols - noOfPixels):
            if DamageSignsegmented[r, c] > 0:
                countPixels = 0
                for i in range(0, noOfPixels):
                    if DamageSignsegmented[r, c + i] > 0:
                        countPixels = countPixels + 1
                if countPixels == noOfPixels:
                    for d in range(0, noOfPixels):
                        DamageSignsegmented[r, c + d] = 0

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignsegmented)

    """
    structuring elements for noise removal radius is based on Moore neighborhood
    """

    def remove_noise_points(radius, minPoints, image):
        labels = np.zeros((VNIRnumOfRows, VNIRnumOfCols))

        segmented = image

        for r in range(0, VNIRnumOfRows):
            for c in range(0, VNIRnumOfCols):
                if segmented[r, c] > 0:
                    labels[r, c] = 1

        for r in range(radius - 1, VNIRnumOfRows - radius):
            for c in range(radius - 1, VNIRnumOfCols - radius):
                if segmented[r, c] > 0:
                    numberOfNieghbors = 0
                    for radRow in range(r - radius + 1, r + radius):
                        for radCol in range(c - radius + 1, c + radius):
                            numberOfNieghbors = numberOfNieghbors + labels[radRow, radCol]
                    if numberOfNieghbors < minPoints:
                        segmented[r, c] = 0


    radius = 3
    minPoints = 10
    iterations = 2
    for i in range(0, iterations):
        remove_noise_points(radius, minPoints, VNIRX1segmented)

    iterations = 1
    for i in range(0, iterations):
        remove_noise_points(radius, minPoints, Y1Y2Y3X2segmented)

    radius = 4
    minPoints = 15
    iterations = 2
    for i in range(0, iterations):
        remove_noise_points(radius, minPoints, DamageSignsegmented)

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRX1segmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", Y1Y2Y3X2segmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignsegmented)

    # //////////////////////////////// find VNIRX1 and VNIRY1Y2Y3X2 and damagesign points //////////////////////////
    for c in range(int(VNIRnumOfCols / 3), 0, -1):
        count = 0
        VNIRX1distance = 0
        for r in range(0, VNIRnumOfRows):
            if VNIRX1segmented[r, c] > 0:
                count = count + 1
        if count >= 4:
            VNIRX1distance = c
            break
    c = 20
    count = 0
    for r in range(0, VNIRnumOfRows):
        if Y1Y2Y3X2segmented[r, c] > 0:
            break
        count = count + 1
    VNIRY1distance = count

    c = VNIRnumOfCols - 20
    count = 0
    for r in range(0, VNIRnumOfRows):
        if Y1Y2Y3X2segmented[r, c] > 0:
            break
        count = count + 1
    VNIRY2distance = count

    DamageSignXdistance = 0
    for r in range(210, 350):
        count = 0
        for c in range(0, VNIRnumOfCols):
            if DamageSignsegmented[r, c] > 0:
                DamageSignXdistance = count # is two points ahead (DamageSignXdistance-2 will give the mid point)
                break
            count = count + 1


    print(VNIRX1distance)
    print(VNIRY1distance)
    print(VNIRY2distance)
    print(DamageSignXdistance) # is two points ahead (DamageSignXdistance-2 will give the mid point)

    # //////////////////////////////////////////// export data as image or excel file //////////////////////////////
    # cv2.imwrite("../../SegmentedImages/" + "/" + "1.png", image[:, :, 10])
    # segSlice1 = np.zeros((imgRows, imgColumns))
    print(str(VNIRimages + 1) + " images of " + str(len(VNIRimagesArray)))

