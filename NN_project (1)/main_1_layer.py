import os
import sys
import io

import os.path
from PIL import Image, ImageFilter, ImageOps
import math
import numpy
import threading

# from sklearn import utils

# longest word in english
string_longest = "Pneumonoultramicroscopicsilicovolcanoconiosis"

ascii_longest = bytearray(string_longest.encode())
i = 0
binary = 0
pow1 = 0
for ascii in ascii_longest:
    n = int(ascii)
    while n > 0:
        binary = binary + ((n % 2) * (pow(10, pow1)))
        pow1 = pow1 + 1
        n = int(n / 2)

y_j = pow1
y_train = numpy.zeros((655, 315))

y_test = numpy.zeros((1965, 315))

x_train = numpy.zeros((655, 11680))

# blurry
x_test = numpy.zeros((1965, 11680))


# save all the images as x and y as labels from a given path
def to_binary_arr(y, label, index_train):
    ascii_longest = label
    num_of_byte = 0
    for ascii in ascii_longest:
        n = int(ascii)
        while n > 0:
            j = 315 - num_of_byte - 1
            y[index_train][j] = n % 2
            num_of_byte = num_of_byte + 1
            n = int(n / 2)


def save_all_data(x_train, y_train, x_test, y_test, current_path, label, index_train, index_test):
    for f in os.listdir(current_path):
        if f != ".DS_Store":
            try:
                name_of_file = int(f.split(".")[0])

            except ValueError:
                save_all_data(x_train, y_train, x_test, y_test, current_path + f + "/", bytearray(f.encode()),
                              index_train, index_test)
                continue
                # in every folder there is 524 photos,
                # so we divide it to 4.
                # 1 group is not blurry and the other 3 are blurry
            if name_of_file <= 131:
                to_binary_arr(y_train, label, index_train)

                image = Image.open(current_path + f)
                gray_image = image.convert('L')
                k = 0
                for i in range(gray_image.width):
                    for j in range(gray_image.height):
                        current_pixel = gray_image.getpixel((i, j))
                        if current_pixel > 120:
                            current_pixel = 0
                        else:
                            current_pixel = 1
                        x_train[index_train][k] = current_pixel
                        k += 1
                index_train += 1

            else:
                to_binary_arr(y_test, label, index_test)
                image = Image.open(current_path + f)
                gray_image = image.filter(ImageFilter.BLUR).convert('L')

                k = 0
                for i in range(gray_image.width):
                    for j in range(gray_image.height):
                        current_pixel = gray_image.getpixel((i, j))
                        if current_pixel > 120:
                            current_pixel = 0
                        else:
                            current_pixel = 1
                        x_test[index_test][k] = current_pixel
                        k += 1
                index_test += 1


path = "C:/Neural Network/data/small/"
save_all_data(x_train, y_train, x_test, y_test, path, "", 0, 0)


x_train = x_train.reshape(655, 146 * 80).astype(int)
x_test = x_test.reshape(1965, 146 * 80).astype(int)


image_max = 80 * 146
num_of_neurons = 16
# all the pictures are the same
# initialize random weights
# dim = resolution * 16
numpy.random.seed(1)
arr_synapse_one_to_two = numpy.random.rand(image_max, num_of_neurons)
# dim = 16 * the longest word
# initialize random weights
arr_synapse_two_to_tree = numpy.random.rand(num_of_neurons, y_j)


# sigmoid function
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def derivative(x):
    return x * (1 - x)


def update_weights(arr_synapse, arr, arr_delta):
    arr_synapse += arr.T.dot(arr_delta)


# 20000 iterations

print("   --------------------------------------------------------")
print()
print("                      TRAINING NETWORK ")
print()
print("   --------------------------------------------------------")
print()
print()

for i in range(20000):
    # forward propagation
    arr_2 = sigmoid(numpy.dot(x_train, arr_synapse_one_to_two))
    arr_3 = sigmoid(numpy.dot(arr_2, arr_synapse_two_to_tree))
    output_error1 = (arr_3 - y_train)**2
    output_error1 = output_error1.mean()

    print("Success rate:" + str((1 - output_error1) * 100) + "%")
    print("Loop number=" + str(i + 1))
    print()
    if (1-output_error1)*100 >= 90:
        arr_2 = sigmoid(numpy.dot(x_test, arr_synapse_one_to_two))
        arr_3 = sigmoid(numpy.dot(arr_2, arr_synapse_two_to_tree))

        output_error1 = (arr_3 - y_test) ** 2
        output_error1 = output_error1.mean()
        print()
        print()
        print("Test accuracy: " + str((1 - output_error1) * 100) + "%")
        print("Finished at loop number: " + str(i + 1))
        break

    # back propagation
    k2_delta = (y_train - arr_3) * derivative(arr_3)

    k1_error = k2_delta.dot(arr_synapse_two_to_tree.T)
    # how much did each k1 value contribute to the k2 error (according to the weights)?

    # in what direction is the target k1?
    # were we really sure? if so, don't change too much.

    k1_delta = k1_error * derivative(arr_2)
    arr_synapse_two_to_tree += 0.01*arr_2.T.dot(k2_delta)
    arr_synapse_one_to_two += 0.01*x_train.T.dot(k1_delta)
