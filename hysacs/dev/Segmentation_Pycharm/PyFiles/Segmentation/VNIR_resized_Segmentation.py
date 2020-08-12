import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi


VNIRFileSource = "../../SegmentedImages/SourceFolder/VNIR.txt"
imageType = "VNIR"
imagesArray = []
with open(VNIRFileSource, "r") as listOfImages:
    for line in listOfImages:
        imagesArray.append(line.strip())

VNIRdataset = np.array([])
# VNIRdataset = np.arange(1, 257)
countImages = 0

for images in range(0, len(imagesArray)):
    imageIndex = images
    imageDirectory = "G:/VNIR/"
    imageName = imagesArray[imageIndex]
    countImages = countImages + 1

    VNIRimage = np.load(imageDirectory + imageName + ".npy")

    # print(image.shape)

    VNIRnumOfRows = VNIRimage.shape[0]
    VNIRnumOfCols = VNIRimage.shape[1]
    VNIRnumOfBands = VNIRimage.shape[2]

    VNIRSegmentchannels1 = np.array([23, 31, 39])
    VNIRSegmentMax1 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels1[0])])
    VNIRSegmentMin1 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels1[0])])
    VNIRSegmentMax2 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels1[1])])
    VNIRSegmentMin2 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels1[1])])
    VNIRSegmentMax3 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels1[2])])
    VNIRSegmentMin3 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels1[2])])

    VNIRSegmentchannels2 = np.array([159, 140, 120])
    VNIRSegmentMax4 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels2[0])])
    VNIRSegmentMin4 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels2[0])])
    VNIRSegmentMax5 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels2[1])])
    VNIRSegmentMin5 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels2[1])])
    VNIRSegmentMax6 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels2[2])])
    VNIRSegmentMin6 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels2[2])])

    VNIRSegmentchannels3 = np.array([150, 150, 115])
    VNIRSegmentMax7 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels3[0])])
    VNIRSegmentMin7 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels3[0])])
    VNIRSegmentMax8 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels3[1])])
    VNIRSegmentMin8 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels3[1])])
    VNIRSegmentMax9 = np.max(VNIRimage[:, :, int(VNIRSegmentchannels3[2])])
    VNIRSegmentMin9 = np.min(VNIRimage[:, :, int(VNIRSegmentchannels3[2])])

    VNIRSegmentslice1 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    VNIRSegmentslice2 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    VNIRSegmentslice3 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))

    VNIRSegmentslice4 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    VNIRSegmentslice5 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    VNIRSegmentslice6 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))

    VNIRSegmentslice7 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    VNIRSegmentslice8 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    VNIRSegmentslice9 = np.zeros((VNIRnumOfRows, VNIRnumOfCols))

    for r in range(0, VNIRnumOfRows):
        if 5 < r < 175:
            for c in range(0, VNIRnumOfCols):
                if 305 > c > 50:
                    VNIRSegmentslice1[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels1[0])] - VNIRSegmentMin1) / \
                                              (VNIRSegmentMax1 - VNIRSegmentMin1) * 255
                    VNIRSegmentslice2[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels1[1])] - VNIRSegmentMin3) / \
                                              (VNIRSegmentMax3 - VNIRSegmentMin3) * 255
                    VNIRSegmentslice3[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels1[2])] - VNIRSegmentMin2) / \
                                              (VNIRSegmentMax2 - VNIRSegmentMin2) * 255
                    VNIRSegmentslice4[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels2[0])] - VNIRSegmentMin4) / \
                                              (VNIRSegmentMax4 - VNIRSegmentMin4) * 255
                    VNIRSegmentslice5[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels2[1])] - VNIRSegmentMin6) / \
                                              (VNIRSegmentMax6 - VNIRSegmentMin6) * 255
                    VNIRSegmentslice6[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels2[2])] - VNIRSegmentMin5) / \
                                              (VNIRSegmentMax5 - VNIRSegmentMin5) * 255
                    # VNIRSegmentslice7[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels3[0])] - VNIRSegmentMin7) / \
                    #                           (VNIRSegmentMax7 - VNIRSegmentMin7) * 255
                    # VNIRSegmentslice8[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels3[1])] - VNIRSegmentMin9) / \
                    #                           (VNIRSegmentMax9 - VNIRSegmentMin9) * 255
                    # VNIRSegmentslice9[r, c] = (VNIRimage[r, c, int(VNIRSegmentchannels3[2])] - VNIRSegmentMin8) / \
                    #                           (VNIRSegmentMax8 - VNIRSegmentMin8) * 255


    VNIRSegmentrgbImage1 = np.zeros([int(VNIRnumOfRows), VNIRnumOfCols, 3], dtype=np.uint8)
    VNIRSegmentrgbImage2 = np.zeros([int(VNIRnumOfRows), VNIRnumOfCols, 3], dtype=np.uint8)
    VNIRSegmentrgbImage3 = np.zeros([int(VNIRnumOfRows), VNIRnumOfCols, 3], dtype=np.uint8)
    for r in range(0, int(VNIRnumOfRows)):
        if 8 < r < 185:
            for c in range(0, VNIRnumOfCols):
                if 305 > c > 50:
                    VNIRSegmentrgbImage1[r, c] = [VNIRSegmentslice1[r, c],
                                                  VNIRSegmentslice2[r, c],
                                                  VNIRSegmentslice3[r, c]]
                    VNIRSegmentrgbImage2[r, c] = [VNIRSegmentslice4[r, c],
                                                  VNIRSegmentslice5[r, c],
                                                  VNIRSegmentslice6[r, c]]
                    # VNIRSegmentrgbImage3[r, c] = [VNIRSegmentslice7[r, c],
                    #                               VNIRSegmentslice8[r, c],
                    #                               VNIRSegmentslice9[r, c]]
    # cv2.imwrite("G:/multiCombinations/segmented/" + imageName + ".png", VNIRSegmentrgbImage1)
    cv2.imwrite("G:/test/" + imageName + "rgb.png", VNIRSegmentrgbImage2)
    # cv2.imwrite("G:/multiCombinations/" + imageName + "2.png", VNIRSegmentrgbImage3)

    # ///////////////////////////////////////////////////// segment grains ///////////////////////////////////////
    # nemo = cv2.imread(imageDirectory + imageName)
    VNIRSegmentsegmentrgb1 = VNIRSegmentrgbImage1
    # plt.imshow(segmentrgb)
    # plt.show()
    VNIRSegmented3d1 = cv2.cvtColor(VNIRSegmentsegmentrgb1, cv2.COLOR_RGB2BGR)
    # plt.imshow(segmented3d)
    # plt.show()
    VNIRSegmenthsvcoverted1 = cv2.cvtColor(VNIRSegmented3d1, cv2.COLOR_RGB2HSV)
    VNIRSegment_lower_bounds1 = (0, 150, 80)
    VNIRSegment_upper_bounds1 = (10, 255, 255)
    VNIRSegmentmask1 = cv2.inRange(VNIRSegmenthsvcoverted1, VNIRSegment_lower_bounds1, VNIRSegment_upper_bounds1)
    VNIRSegmentresult1 = cv2.bitwise_and(VNIRSegmented3d1, VNIRSegmented3d1, mask=VNIRSegmentmask1)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    VNIRSegmentresult1 = cv2.cvtColor(VNIRSegmentresult1, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("G:/multiCombinations/" + imageName + "2.png", VNIRSegmentresult1)

    # nemo = cv2.imread(imageDirectory + imageName)
    VNIRSegmentsegmentrgb2 = VNIRSegmentrgbImage2
    # plt.imshow(segmentrgb)
    # plt.show()
    VNIRSegmented3d2 = cv2.cvtColor(VNIRSegmentsegmentrgb2, cv2.COLOR_RGB2BGR)
    # plt.imshow(segmented3d)
    # plt.show()
    VNIRSegmenthsvcoverted2 = cv2.cvtColor(VNIRSegmented3d2, cv2.COLOR_RGB2HSV)
    VNIRSegment_lower_bounds2 = (0, 0, 0)
    VNIRSegment_upper_bounds2 = (50, 255, 255)
    VNIRSegmentmask2 = cv2.inRange(VNIRSegmenthsvcoverted2, VNIRSegment_lower_bounds2, VNIRSegment_upper_bounds2)
    VNIRSegmentresult2 = cv2.bitwise_and(VNIRSegmented3d2, VNIRSegmented3d2, mask=VNIRSegmentmask2)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    VNIRSegmentresult2 = cv2.cvtColor(VNIRSegmentresult2, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("G:/multiCombinations/" + imageName + "3.png", VNIRSegmentresult2)

    # nemo = cv2.imread(imageDirectory + imageName)
    VNIRSegmentsegmentrgb3 = VNIRSegmentrgbImage3
    # plt.imshow(segmentrgb)
    # plt.show()
    VNIRSegmented3d3 = cv2.cvtColor(VNIRSegmentsegmentrgb3, cv2.COLOR_RGB2BGR)
    # plt.imshow(segmented3d)
    # plt.show()
    VNIRSegmenthsvcoverted3 = cv2.cvtColor(VNIRSegmented3d3, cv2.COLOR_RGB2HSV)
    VNIRSegment_lower_bounds3 = (50, 0, 0) # 30,
    VNIRSegment_upper_bounds3 = (70, 255, 255) # 60,
    VNIRSegmentmask3 = cv2.inRange(VNIRSegmenthsvcoverted3, VNIRSegment_lower_bounds3, VNIRSegment_upper_bounds3)
    VNIRSegmentresult3 = cv2.bitwise_and(VNIRSegmented3d3, VNIRSegmented3d3, mask=VNIRSegmentmask3)
    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    VNIRSegmentresult3 = cv2.cvtColor(VNIRSegmentresult3, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("G:/multiCombinations/segmented/missing/" + imageName + "4.png", VNIRSegmentresult3)

    # //////////////////////////////////// Joining Two rgb (6 channels) images //////////////////////////////////

    VNIRsegmented = np.zeros((VNIRnumOfRows, VNIRnumOfCols))
    for r in range(0, VNIRnumOfRows):
        for c in range(0, VNIRnumOfCols):
            if VNIRSegmentresult1[r, c, 0] or VNIRSegmentresult1[r, c, 1] or \
                    VNIRSegmentresult1[r, c, 2] or VNIRSegmentresult2[r, c, 0] or \
                    VNIRSegmentresult2[r, c, 1] or VNIRSegmentresult2[r, c, 2]:
                VNIRsegmented[r, c] = VNIRimage[r, c, 159]

    VNIRSegmentmaxx = np.max(VNIRsegmented[:, :])
    VNIRSegmentMinx = np.min(VNIRsegmented[:, :])
    for r in range(0, VNIRnumOfRows):
        if 8 < r < 165:
            for c in range(0, VNIRnumOfCols):
                if 305 > c > 50:
                    VNIRsegmented[r, c] = (VNIRsegmented[r, c] - VNIRSegmentMinx) / \
                                              (VNIRSegmentmaxx - VNIRSegmentMinx) * 255

    # cv2.imwrite("G:/multiCombinations/" + imageName + ".png", VNIRsegmented)

    # ///////////////////////////////////////////////// structuring elements //////////////////////////////////
    """
        applying structuring elements for noise removal radius is based on Moore neighborhood
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


    # cleaning grains
    radius = 6
    minPoints = 48
    iterations = 1
    for i in range(0, iterations):
        remove_noise_points(radius, minPoints, VNIRsegmented)

    # radius = 4
    # minPoints = 32
    # iterations = 3
    # for i in range(0, iterations):
    #     remove_noise_points(radius, minPoints)

    # cv2.imwrite("G:/multiCombinations/" + imageName + ".png", VNIRsegmented)


    # ///////////////////////////////////////////// get middle grains of each spike ///////////////////////////////

    def selectpixels(midvertical, midhorizontal, radius, image):
        numberofpoints = 0
        minimumpoints = 1000
        expansionstep = 2
        VNIRSegmented = image
        for r in range(-radius, radius):
            for c in range(-radius, radius):
                if VNIRSegmented[midvertical + r, midhorizontal + c] > 0:
                    numberofpoints = numberofpoints + 1
                    VNIRSegmented[midvertical + r, midhorizontal + c] = 255
        for r in range(0, 55):
            if numberofpoints < minimumpoints and radius < 154:
                radius = radius + expansionstep
                numberofpoints = 0
                for r in range(-radius, radius):
                    for c in range(-radius, radius):
                        if VNIRSegmented[midvertical + r, midhorizontal + c] > 0:
                            numberofpoints = numberofpoints + 1
                            VNIRSegmented[midvertical + r, midhorizontal + c] = 255

        # print(radius)
        # print(numberofpoints)

    selectpixels(100, 190, 20, VNIRsegmented)

    # cv2.imwrite("G:/test/" + imageName + ".png", VNIRsegmented)

    # ///////////////////////////////////// preparing dataset to build AI models pixel_wise ////////////////////////

    # imageNameShort = ""
    # for imgIndex in range(0,6):
    #     imageNameShort = imageNameShort + imageName[imgIndex]
    # imageNameShort = int(imageNameShort)
    #
    # VNIRdataset = np.array([])
    # pixels_position_spectral = np.zeros((VNIRnumOfBands + 3, 1))
    #
    # countNumberOfPixels = 0
    # for r in range(0, VNIRnumOfRows):
    #     for c in range(0, VNIRnumOfCols):
    #         if VNIRsegmented[r, c] == 255:
    #             count = 0
    #             for a in range(0, VNIRnumOfBands + 2):
    #                 countNumberOfPixels = countNumberOfPixels + 1
    #                 count = count + 1
    #                 if count == 1:
    #                     pixels_position_spectral[a] = imageNameShort
    #                 if count == 2:
    #                     pixels_position_spectral[a] = r
    #                 if count == 3:
    #                     pixels_position_spectral[a] = c
    #                 if count > 3:
    #                     pixels_position_spectral[a] = VNIRimage[r, c, a - 2]
    #             VNIRdataset = np.append(VNIRdataset, pixels_position_spectral)
    # for r in range(0, VNIRnumOfRows):
    #     for c in range(0, VNIRnumOfCols):
    #         if VNIRsegmented[r, c] == 255:
    #             print(VNIRimage[r, c, 0])
    #
    # # test result
    # # print(pixels_position_spectral)
    # # print(VNIRimage[584, 147, :])
    # # VNIRdataset = np.append(VNIRdataset, pixels_position_spectral)
    # # VNIRdataset = np.reshape(VNIRdataset, (int(countNumberOfPixels/(VNIRnumOfBands + 3)+7), VNIRnumOfBands + 3))
    # np.save("G:/dataset/" + imageName, VNIRdataset, allow_pickle=True, fix_imports=True)
    # np.savetxt("G:/dataset/" + imageName + ".csv", VNIRdataset, delimiter=',')

    # /////////////////////////////////////// preparing dataset based on mean and std.dev //////////////////////////

    # VNIRvariablesName = np.arange(1, 266)
    # VNIRdataset = np.zeros((VNIRnumOfBands, 3))
    VNIRmean = np.zeros((VNIRnumOfBands, 1))
    VNIRstd_dev = np.zeros((VNIRnumOfBands, 1))

    for b in range(0, VNIRnumOfBands):
        count = 0
        sum = 0
        meanValue = 0
        for r in range(0, VNIRnumOfRows):
            if 5 < r < 175:
                for c in range(0, VNIRnumOfCols):
                    if VNIRsegmented[r, c] == 255:
                        count = count + 1
                        sum = sum + VNIRimage[r, c, b]
        meanValue = sum / count
        VNIRmean[b] = meanValue

    for b in range(0, VNIRnumOfBands):
        count = 0
        std_dev = 0
        variance = 0
        sum = 0
        for r in range(0, VNIRnumOfRows):
            if 5 < r < 175:
                for c in range(0, VNIRnumOfCols):
                    if VNIRsegmented[r, c] == 255:
                        count = count + 1
                        sum = sum + (VNIRimage[r, c, b] - VNIRmean[b]) * (VNIRimage[r, c, b] - VNIRmean[b])
        variance = sum / count
        std_dev = np.math.sqrt(variance)
        VNIRstd_dev[b] = std_dev

    VNIRSingleImagedataset = np.concatenate((VNIRmean, VNIRstd_dev))
    VNIRdataset = np.append(VNIRdataset, VNIRSingleImagedataset)

    # //////////////////////////////////////////// export data as image or excel file //////////////////////////////
    # cv2.imwrite("G:/IntactVNIR/" + imageName + ".png", image[:, :, 10])
    # segSlice1 = np.zeros((imgRows, imgColumns))
    print(str(images + 1) + " images of " + str(len(imagesArray)))

# ////////////////////////////// export dataset based on mean to excel file and numpy array ////////////////////////

VNIRdataset = np.reshape(VNIRdataset, (countImages * 2, VNIRnumOfBands))

VNIRdataset = VNIRdataset.transpose()

np.save("G:/dataset/" + "VNIR_dataset_basedon_mean_stddev", VNIRdataset, allow_pickle=True, fix_imports=True)
np.savetxt("G:/dataset/" + "VNIR_dataset_basedon_mean_stddev.csv", VNIRdataset, delimiter=',')

# ////////////////////////////// export dataset pixel wise to excel file and numpy array ////////////////////////
# VNIRdataset = np.reshape(VNIRdataset, (countImages * 1000, 259))
#
# np.save("G:/dataset/" + "VNIR_dataset_pixelwise", VNIRdataset, allow_pickle=True, fix_imports=True)
# np.savetxt("G:/dataset/" + "VNIR_dataset_pixelwise.csv", VNIRdataset, delimiter=',')

