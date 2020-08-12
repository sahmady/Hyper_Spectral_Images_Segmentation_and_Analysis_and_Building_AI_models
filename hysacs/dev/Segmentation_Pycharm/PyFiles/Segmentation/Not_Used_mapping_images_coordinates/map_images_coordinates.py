import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi

""" 
Merging both Images by mapping coordinates of images
    Note! we haven't proceeded with this approach and instead we merged extracted 
    dataset of each image type which provided based on mean values
"""

# ///////////////////////////////////// VNIR and SWIR images source /////////////////////////////////////////
VNIRFileSource = "../../SegmentedImages/SourceFolder/VNIR.txt"
VNIRimageType = "VNIR"
VNIRimagesArray = []
with open(VNIRFileSource, "r") as VNIRlistOfImages:
    for line in VNIRlistOfImages:
        VNIRimagesArray.append(line.strip())

SWIRFileSource = "../../SegmentedImages/SourceFolder/SWIR.txt"
SWIRimageType = "SWIR"
SWIRimagesArray = []
with open(SWIRFileSource, "r") as SWIRlistOfImages:
    for line in SWIRlistOfImages:
        SWIRimagesArray.append(line.strip())

for mutualExclusiveImageList in range(0, len(VNIRimagesArray)):
    VNIRimageIndex = mutualExclusiveImageList
    VNIRimageDirectory = "G:/VNIR/"
    VNIRimageName = VNIRimagesArray[VNIRimageIndex]

    SWIRimageIndex = mutualExclusiveImageList
    SWIRimageDirectory = "F:/SWIR/"
    SWIRimageName = SWIRimagesArray[SWIRimageIndex]

    # ///////////////////////////////////////// read/load images /////////////////////////////////////////////
    VNIRimage = np.load(VNIRimageDirectory + VNIRimageName + ".npy")
    SWIRimage = envi.open(SWIRimageDirectory + SWIRimageName + '.hdr',
                          SWIRimageDirectory + SWIRimageName + '.img')

    VNIRnumOfRows = VNIRimage.shape[0]
    VNIRnumOfCols = VNIRimage.shape[1]
    VNIRnumOfBands = VNIRimage.shape[2]

    SWIRnumOfRows = SWIRimage.nrows
    SWIRnumOfCols = SWIRimage.ncols
    SWIRnumOfBands = SWIRimage.nbands

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

    VNIRSegmentchannels = np.array([23, 31, 39])
    VNIRSegmentMax1 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels[0])])
    VNIRSegmentMin1 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels[0])])
    VNIRSegmentMax2 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels[1])])
    VNIRSegmentMin2 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels[1])])
    VNIRSegmentMax3 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels[2])])
    VNIRSegmentMin3 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels[2])])

    VNIRX1channelsslice1 = np.zeros((int(VNIRnumOfRows / 4), int(VNIRnumOfCols / 3)))
    VNIRX1channelsslice2 = np.zeros((int(VNIRnumOfRows / 4), int(VNIRnumOfCols / 3)))
    VNIRX1channelsslice3 = np.zeros((int(VNIRnumOfRows / 4), int(VNIRnumOfCols / 3)))

    VNIRY1Y2Y3X2slice1 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))
    VNIRY1Y2Y3X2slice2 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))
    VNIRY1Y2Y3X2slice3 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))

    VNIRDamageSignslice1 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))
    VNIRDamageSignslice2 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))
    VNIRDamageSignslice3 = np.zeros((int(VNIRnumOfRows), int(VNIRnumOfCols)))

    VNIRSegmentslice1 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    VNIRSegmentslice2 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    VNIRSegmentslice3 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))

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

    for r in range(0, VNIRnumOfRows):
        # if r < 210:
            for c in range(0, VNIRnumOfCols):
                VNIRSegmentslice1[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels[0])] - VNIRSegmentMin1) / \
                                          (VNIRSegmentMax1 - VNIRSegmentMin1) * 255
                VNIRSegmentslice2[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels[1])] - VNIRSegmentMin3) / \
                                          (VNIRSegmentMax3 - VNIRSegmentMin3) * 255
                VNIRSegmentslice3[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels[2])] - VNIRSegmentMin2) / \
                                          (VNIRSegmentMax2 - VNIRSegmentMin2) * 255


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

    VNIRSegmentrgbImage = np.zeros([int(VNIRnumOfRows), VNIRnumOfCols, 3], dtype=np.uint8)
    for r in range(0, int(VNIRnumOfRows)):
        for c in range(0, VNIRnumOfCols):
            VNIRSegmentrgbImage[r, c] = [VNIRSegmentslice1[r, c], VNIRSegmentslice2[r, c], VNIRSegmentslice3[r, c]]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRX1channelsrgbImage)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRY1Y2Y3X2rgbImage)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRDamageSignrgbImage)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", rgbImage)

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

    SWIRSegmentchannels = np.array([55, 41, 12])
    SWIRSegmentMax1 = np.max(SWIRimage[:, :, int(SWIRSegmentchannels[0])])
    SWIRSegmentMin1 = np.min(SWIRimage[:, :, int(SWIRSegmentchannels[0])])
    SWIRSegmentMax2 = np.max(SWIRimage[:, :, int(SWIRSegmentchannels[1])])
    SWIRSegmentMin2 = np.min(SWIRimage[:, :, int(SWIRSegmentchannels[1])])
    SWIRSegmentMax3 = np.max(SWIRimage[:, :, int(SWIRSegmentchannels[2])])
    SWIRSegmentMin3 = np.min(SWIRimage[:, :, int(SWIRSegmentchannels[2])])

    SWIRX1channelsslice1 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    SWIRX1channelsslice2 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    SWIRX1channelsslice3 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))

    SWIRY1Y2Y3X2slice1 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    SWIRY1Y2Y3X2slice2 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    SWIRY1Y2Y3X2slice3 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))

    # SWIRDamageSignslice1 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    # SWIRDamageSignslice2 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))
    # SWIRDamageSignslice3 = np.zeros((int(SWIRnumOfRows), int(SWIRnumOfCols)))

    SWIRSegmentslice1 = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    SWIRSegmentslice2 = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    SWIRSegmentslice3 = np.zeros((SWIRnumOfRows, SWIRnumOfCols))

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

    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                SWIRSegmentslice1[r, c] = (SWIRimage[r, c, int(SWIRSegmentchannels[0])] - SWIRSegmentMin1) /\
                                          (SWIRSegmentMax1 - SWIRSegmentMin1) * 255
                SWIRSegmentslice2[r, c] = (SWIRimage[r, c, int(SWIRSegmentchannels[1])] - SWIRSegmentMin3) /\
                                          (SWIRSegmentMax3 - SWIRSegmentMin3) * 255
                SWIRSegmentslice3[r, c] = (SWIRimage[r, c, int(SWIRSegmentchannels[2])] - SWIRSegmentMin2) /\
                                          (SWIRSegmentMax2 - SWIRSegmentMin2) * 255

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


    SWIRSegmentimage = np.zeros([SWIRnumOfRows, SWIRnumOfCols, 3], dtype=np.uint8)
    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                SWIRSegmentimage[r, c] = [SWIRSegmentslice1[r, c],
                                          SWIRSegmentslice2[r, c],
                                          SWIRSegmentslice3[r, c]]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRX1channelsrgbImage)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRY1Y2Y3X2rgbImage)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRDamageSignrgbImage)
    # cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", image)

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

    # nemo = cv2.imread(imageDirectory + imageName)
    VNIRSegmentsegmentrgb = VNIRSegmentrgbImage
    # plt.imshow(segmentrgb)
    # plt.show()
    VNIRSegmented3d = cv2.cvtColor(VNIRSegmentsegmentrgb, cv2.COLOR_RGB2BGR)
    # plt.imshow(segmented3d)
    # plt.show()
    VNIRSegmenthsvcoverted = cv2.cvtColor(VNIRSegmented3d, cv2.COLOR_RGB2HSV)
    VNIRSegment_lower_bounds = (0, 0, 80)  # light_range = (0, 0, 80) works perfectly for 23, 31, 39 channels
    VNIRSegment_upper_bounds = (10, 250, 255)  # (10, 250, 255)
    VNIRSegmentmask = cv2.inRange(VNIRSegmenthsvcoverted, VNIRSegment_lower_bounds, VNIRSegment_upper_bounds)
    VNIRSegmentresult = cv2.bitwise_and(VNIRSegmented3d, VNIRSegmented3d, mask=VNIRSegmentmask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    VNIRSegmentresult = cv2.cvtColor(VNIRSegmentresult, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRSegmentresult)

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

    # nemo = cv2.imread("../../SegmentedImages/" + imageType + "/" + imageName + ".png")
    SWIRSegmentedrgb = SWIRSegmentimage
    # plt.imshow(segmentrgb)
    # plt.show()
    SWIRSegmented3d = cv2.cvtColor(SWIRSegmentedrgb, cv2.COLOR_BGR2RGB)
    # plt.imshow(segmented3d)
    # plt.show()
    SWIRSegmenthsvcoverted = cv2.cvtColor(SWIRSegmented3d, cv2.COLOR_RGB2HSV)
    SWIRSegment_lower_bounds = (0, 60, 50)
    SWIRSegment_upper_bounds = (100, 255, 255)
    SWIRSegmentmask = cv2.inRange(SWIRSegmenthsvcoverted, SWIRSegment_lower_bounds, SWIRSegment_upper_bounds)
    SWIRSegmentresult = cv2.bitwise_and(SWIRSegmented3d, SWIRSegmented3d, mask=SWIRSegmentmask)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    SWIRSegmentresult = cv2.cvtColor(SWIRSegmentresult, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("../../SegmentedImages/" + imageType + "/" + imageName + ".png", result)

    # ////////////////////////////////////// keep only one slice of VNIR image ///////////////////////////////////

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

    VNIRsegmented = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    for r in range(0, VNIRnumOfRows):
        for c in range(0, VNIRnumOfCols):
            if VNIRSegmentresult[r, c, 0] or VNIRSegmentresult[r, c, 1] or VNIRSegmentresult[r, c, 2]:
                VNIRsegmented[r, c] = VNIRSegmentresult[r, c, 2]

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRsegmented)

    # ////////////////////////////////////// keep only one slice of SWIR image ///////////////////////////////

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

    SWIRSegmented = np.zeros((SWIRnumOfRows, SWIRnumOfCols))
    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                if SWIRSegmentresult[r, c, 0] or SWIRSegmentresult[r, c, 1] or SWIRSegmentresult[r, c, 2]:
                    SWIRSegmented[r, c] = SWIRSegmentresult[r, c, 2]

    # /////////////////////////////////////// removing lines zero degrees of VNIR image //////////////////////////

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

    # /////////////////////////////////////// removing lines zero degrees of SWIR image //////////////////////////

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

    # ////////////////////////////////////////// quadratic transformation SWIR //////////////////////////////////
    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                if SWIRSegmented[r, c] > 0:
                    SWIRSegmented[r, c] = SWIRSegmented[r, c] * SWIRSegmented[r, c]

    SWIRSegmentedmaxValue = np.max(SWIRSegmented[:, :])
    SWIRSegmentedminValue = np.min(SWIRSegmented[:, :])

    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                SWIRSegmented[r, c] = (SWIRSegmented[r, c] - SWIRSegmentedminValue) / \
                                      (SWIRSegmentedmaxValue - SWIRSegmentedminValue) * 250

    for r in range(0, SWIRnumOfRows):
        if 630 > r > 420:
            for c in range(0, SWIRnumOfCols):
                if SWIRSegmented[r, c] < 30:
                    SWIRSegmented[r, c] = 0

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


    # /////////////////////////////////////// structuring elements for VNIR image ///////////////////////////////

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

    radius = 5
    minPoints = 20
    iterations = 2
    for i in range(0, iterations):
        remove_noise_points(radius, minPoints, VNIRsegmented)

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRX1segmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", Y1Y2Y3X2segmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", DamageSignsegmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRsegmented)

    # /////////////////////////////////////// structuring elements SWIR image ///////////////////////////////

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

    radius = 5
    minPoints = 50
    iterations = 1
    for i in range(0, iterations):
        remove_noise_points(radius, minPoints, SWIRSegmented)

    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRX1segmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRY1Y2Y3X2segmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRDamageSignsegmented)

    # ///////////////////////////////////////////// get middle grains of each spike ///////////////////////////////

    def selectpixels(midvertical, midhorizontal, radius, image):
        numberofpoints = 0
        minimumpoints = 2000
        expansionstep = 5
        SWIRSegmented = image
        for r in range(-radius, radius):
            for c in range(-radius, radius):
                if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                    numberofpoints = numberofpoints + 1
                    SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255
        if numberofpoints < minimumpoints and radius < 154:
            radius = radius + expansionstep
            numberofpoints = 0
            for r in range(-radius, radius):
                for c in range(-radius, radius):
                    if SWIRSegmented[midvertical + r, midhorizontal + c] > 0:
                        numberofpoints = numberofpoints + 1
                        SWIRSegmented[midvertical + r, midhorizontal + c] = 255

        # print(radius)
        # print(numberofpoints)


    # //////////////////////////////////////// get middle grains of each spike VNIR ///////////////////////////////
    selectpixels(95, 175, 30, VNIRsegmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", VNIRsegmented)

    # //////////////////////////////////////// get middle grains of each spike SWIR ///////////////////////////////
    selectpixels(525, 160, 30, SWIRSegmented)
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", SWIRSegmented)

    # //////////////////////////////// find VNIRX1 and VNIRY1Y2Y3X2 and damagesign points //////////////////////////

    VNIRX1distance = 0
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

    # ///////////////////////////////////////// variable for further implementation //////////////////////////////
    # VNIRsegmented
    # VNIRX1distance
    # VNIRY1distance
    # VNIRY2distance

    # SWIRSegmented
    # SWIRX1distance
    # SWIRY1distance
    # SWIRY2distance

    # //////////////////////////////////////////// export data as image or excel file //////////////////////////////
    # cv2.imwrite("../../SegmentedImages/" + "/" + "1.png", image[:, :, 10])
    # segSlice1 = np.zeros((imgRows, imgColumns))
    print(str(mutualExclusiveImageList + 1) + " images of " + str(len(VNIRimagesArray)))

