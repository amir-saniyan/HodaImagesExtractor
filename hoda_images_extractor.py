# *-* coding: utf-8 *-*

import os
import struct
import numpy as np
import imageio

# Hoda Images Extractor
# Python code for extracting Hoda dataset images.

# Hoda Farsi Digit Dataset:
# http://farsiocr.ir/
# http://farsiocr.ir/مجموعه-داده/مجموعه-ارقام-دستنویس-هدی
# http://dadegan.ir/catalog/hoda

# Repository:
# https://github.com/amir-saniyan/HodaImagesExtractor


def read_hoda_cdb(file_name):
    with open(file_name, 'rb') as binary_file:

        data = binary_file.read()

        offset = 0

        # read private header

        yy = struct.unpack_from('H', data, offset)[0]
        offset += 2

        m = struct.unpack_from('B', data, offset)[0]
        offset += 1

        d = struct.unpack_from('B', data, offset)[0]
        offset += 1

        H = struct.unpack_from('B', data, offset)[0]
        offset += 1

        W = struct.unpack_from('B', data, offset)[0]
        offset += 1

        TotalRec = struct.unpack_from('I', data, offset)[0]
        offset += 4

        LetterCount = struct.unpack_from('128I', data, offset)
        offset += 128 * 4

        imgType = struct.unpack_from('B', data, offset)[0]  # 0: binary, 1: gray
        offset += 1

        Comments = struct.unpack_from('256c', data, offset)
        offset += 256 * 1

        Reserved = struct.unpack_from('245c', data, offset)
        offset += 245 * 1

        if (W > 0) and (H > 0):
            normal = True
        else:
            normal = False

        images = []
        labels = []

        for i in range(TotalRec):

            StartByte = struct.unpack_from('B', data, offset)[0]  # must be 0xff
            offset += 1

            label = struct.unpack_from('B', data, offset)[0]
            offset += 1

            if not normal:
                W = struct.unpack_from('B', data, offset)[0]
                offset += 1

                H = struct.unpack_from('B', data, offset)[0]
                offset += 1

            ByteCount = struct.unpack_from('H', data, offset)[0]
            offset += 2

            image = np.zeros(shape=[H, W], dtype=np.uint8)

            if imgType == 0:
                # Binary
                for y in range(H):
                    bWhite = True
                    counter = 0
                    while counter < W:
                        WBcount = struct.unpack_from('B', data, offset)[0]
                        offset += 1
                        # x = 0
                        # while x < WBcount:
                        #     if bWhite:
                        #         image[y, x + counter] = 0  # Background
                        #     else:
                        #         image[y, x + counter] = 255  # ForeGround
                        #     x += 1
                        if bWhite:
                            image[y, counter:counter + WBcount] = 0  # Background
                        else:
                            image[y, counter:counter + WBcount] = 255  # ForeGround
                        bWhite = not bWhite  # black white black white ...
                        counter += WBcount
            else:
                # GrayScale mode
                data = struct.unpack_from('{}B'.format(W * H), data, offset)
                offset += W * H
                image = np.asarray(data, dtype=np.uint8).reshape([W, H]).T

            images.append(image)
            labels.append(label)

        return images, labels


# Train set.
images, labels = read_hoda_cdb('./DigitDB/Train 60000.cdb')
for i in range(len(images)):
    image = images[i]
    label = labels[i]
    directory_name = './images/train/' + str(label)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_name = directory_name + '/' + str(i) + '.png'
    print('Saving', file_name, '...')
    imageio.imwrite(file_name, image)

# Test set.
images, labels = read_hoda_cdb('./DigitDB/Test 20000.cdb')
for i in range(len(images)):
    image = images[i]
    label = labels[i]
    directory_name = './images/test/' + str(label)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_name = directory_name + '/' + str(i) + '.png'
    print('Saving', file_name, '...')
    imageio.imwrite(file_name, image)

# Remaining samples.
images, labels = read_hoda_cdb('./DigitDB/RemainingSamples.cdb')
for i in range(len(images)):
    image = images[i]
    label = labels[i]
    directory_name = './images/remaining/' + str(label)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    file_name = directory_name + '/' + str(i) + '.png'
    print('Saving', file_name, '...')
    imageio.imwrite(file_name, image)

print('OK')
