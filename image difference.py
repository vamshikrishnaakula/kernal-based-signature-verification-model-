import cv2
import imagehash
import numpy as np
import os
from os import listdir
from pylab import *
from PIL import Image
from scipy.cluster.vq import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import tree, svm, linear_model
from sklearn.metrics import confusion_matrix
from imageio import imread
from sklearn import tree, svm, linear_model
from sklearn.linear_model import LogisticRegression
import pickle
from matplotlib import pyplot as plt

des_list = []
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
im_contour_features = []
display = False
genuine_images_path = ("D:/model/genuine")
forged_images_path = ("D:/model/forged")
genuine_images_path = ("C:/Users/DELL/Downloads/HSig260modified/genuine")
forged_images_path = ("C:/Users/DELL/Downloads/HSig260modified/forged")
genuine_images_path = ("C:/Users/DELL/Desktop/gen/genuine")
forged_images_path = ("C:/Users/DELL/Desktop/for/forged")


genuine_image_filenames = listdir(genuine_images_path)
# print(genuine_image_filenames)
forged_image_filenames = listdir(forged_images_path)
print(forged_image_filenames)
genuine_image_features = [[] for i in range(120)]
forged_image_features = [[] for i in range(120)]
for name in genuine_image_filenames:
    print("name", name)
    signature_id = int(name.split('_')[0][-3:])
    print(signature_id)
    genuine_image_features[signature_id - 1].append({"name": name})
print("genuine_image_features", genuine_image_features)

for name in forged_image_filenames:
    signature_id = int(name.split('_')[0][-3:])
    forged_image_features[signature_id - 1].append({"name": name})

print('the forged_image_features are---------->', forged_image_features)


def preprocess_image(image_path, display=False):
    raw_image = imread(image_path)
    if len(raw_image.shape) < 3:
        bw_image = 255 - raw_image
    else:
        bw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        bw_image = 255 - bw_image

    if display:
        plt.figure(figsize=[8, 8])
        plt.imshow(bw_image, cm='gray')
        plt.axis('off')
        plt.show()

    _, threshold_image = cv2.threshold(bw_image, 30, 255, 0)

    if display:
        plt.figure(figsize=[8, 8])
        plt.imshow(threshold_image, cmap='gray')
        plt.axis('off')
        plt.show()

    return threshold_image


def get_contour_features(preprocessed_image, display=False):
    rect = cv2.minAreaRect(cv2.findNonZero(preprocessed_image))
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])

    aspect_ratio = max(w, h) / min(w, h)
    bounding_rect_area = w * h

    if display:
        image1 = cv2.drawContours(preprocessed_image.copy(), [box], 0, (120, 120, 120), 2)
        cv2.imshow("a", cv2.resize(image1, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    hull = cv2.convexHull(cv2.findNonZero(preprocessed_image))

    if display:
        convex_hull_image = cv2.drawContours(preprocessed_image.copy(), [hull], 0, (120, 120, 120), 2)
        cv2.imshow("a", cv2.resize(convex_hull_image, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()

    _, contours, hierarchy = cv2.findContours(preprocessed_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        contour_image = cv2.drawContours(preprocessed_image.copy(), contours, -1, (120, 120, 120), 3)
        cv2.imshow("a", cv2.resize(contour_image, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey()
    contour_area = 0
    for cnt in contours:
        contour_area += cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull)

    return aspect_ratio, bounding_rect_area, hull_area, contour_area


def sift(preprocessed_image, image_path, display=False):
    raw_image = cv2.imread(image_path)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(preprocessed_image, None)

    if display:
        cv2.drawKeypoints(preprocessed_image, kp, raw_image)
        plt.figure(figsize=[8, 8])
        plt.imshow(cv2.resize(raw_image, (0, 0), fx=3, fy=3))
        plt.axis('off')
        plt.show()

    return (image_path, des)
def features(image_path,display=False):
    if image_path == genuine_images_path:
       for i in range(120):
          for im in genuine_image_features[i]:
             image_path = genuine_images_path +"/"+ im['name']
             print("image_path",image_path)
             preprocessed_image = preprocess_image(image_path)
             print(preprocessed_image)
             hash = imagehash.phash(Image.open(image_path))
             print(hash)
             aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = get_contour_features(
             preprocessed_image.copy(), display=display)
             hash = int(str(hash), 16)
             print("image hash is", hash)
             im['hash'] = hash
             print(hash)
             im['aspect_ratio'] = aspect_ratio
             print(aspect_ratio)
             im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
             print("convex_hull_area / bounding_rect_area", convex_hull_area / bounding_rect_area)
             im['contour_area/bounding_area'] = contours_area / bounding_rect_area
             print("contours_area / bounding_rect_area", contours_area / bounding_rect_area)
             im_contour_features.append([hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])
             print("im_contour_features==", im_contour_features)
             des_list.append(sift(preprocessed_image, image_path))
             print("des_list", des_list)
             print("length of the des_list", len(des_list))
             return des_list

    else:
        for i in range(120):
            for im in forged_image_features[i]:
                    image_path = forged_images_path + "/" + im['name']
                    print("image_path forged",image_path)
                    preprocessed_image = preprocess_image(image_path)
                    print(preprocessed_image)
                    hash = imagehash.phash(Image.open(image_path))
                    print(hash)
                    aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = get_contour_features(
                        preprocessed_image.copy(), display=display)
                    hash = int(str(hash), 16)
                    print("image hash is", hash)
                    im['hash'] = hash
                    print(hash)
                    im['aspect_ratio'] = aspect_ratio
                    print(aspect_ratio)
                    im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
                    print("convex_hull_area / bounding_rect_area", convex_hull_area / bounding_rect_area)
                    im['contour_area/bounding_area'] = contours_area / bounding_rect_area
                    print("contours_area / bounding_rect_area", contours_area / bounding_rect_area)
                    im_contour_features.append(
                        [hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])
                    print("im_contour_features==", im_contour_features)
                    des_list.append(sift(preprocessed_image, image_path))
                    #print("des_list", des_list)
                    print("length of the des_list", len(des_list))
                    return des_list
features(genuine_images_path)
features(forged_images_path)
for i in range(120):
        # print('Learning ' + str(i + 1) + 'th image')
        des_list = []
        for im in genuine_image_features[i]:
            print("genuine_image_features[i]==", len(genuine_image_features[i]))
            print("im======", im)
            print("genuine_image_features[i]===", genuine_image_features[i])
            image_path = genuine_images_path + "/" + im['name']
            print("im['name']====", im['name'])
            print('the image_path is---------->', image_path)
            preprocessed_image = preprocess_image(image_path)
            print(preprocessed_image)
            hash = imagehash.phash(Image.open(image_path))
            print(hash)
            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = get_contour_features(
                preprocessed_image.copy(), display=display)

            hash = int(str(hash), 16)
            print("image hash is", hash)
            im['hash'] = hash
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area

            im_contour_features.append(
                [hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

            des_list.append(sift(preprocessed_image, image_path))
            print("des_list", des_list)
            print(" length of the des_list", len(des_list))

        if len(des_list) == 0:
            #             print(image_path+' does not have suffiecient features, skipping')
            pass

        for im in forged_image_features[i]:
            print("length of the forged_image_features[i]==", len(forged_image_features[i]))
            image_path = forged_images_path + "/" + im['name']
            preprocessed_image = preprocess_image(image_path)
            print("pre process_image", preprocessed_image)
            hash = imagehash.phash(Image.open(image_path))

            aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = get_contour_features(
                preprocessed_image.copy(), display=display)

            hash = int(str(hash), 16)
            im['hash'] = hash
            im['aspect_ratio'] = aspect_ratio
            im['hull_area/bounding_area'] = convex_hull_area / bounding_rect_area
            im['contour_area/bounding_area'] = contours_area / bounding_rect_area

            im_contour_features.append(
                [hash, aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area])

            des_list.append(sift(preprocessed_image, image_path))
            print("des_list", des_list)
            print("des_list===", len(des_list))

        if len(des_list) == 0:
            # print(image_path+' does not have suffiecient features, skipping')
            continue
        descriptors = des_list[0][1]

        print("descriptors", descriptors)
        print("descriptors", len(descriptors))
        for image_path, descriptor in des_list[1:]:
            descriptors = np.vstack((descriptors, descriptor))
        k = 170
        voc, variance = kmeans(descriptors, k, 1)
        # print("voc",voc)
        # print("variance",variance)

        # Calculate the histogram of features
        im_features = np.zeros((len(genuine_image_features[i]) + len(forged_image_features[i]), k + 4), "float32")
        print("im_features==", im_features.shape)
        print("len(genuine_image_features[i])", len(genuine_image_features[i]))
        print("len(forged_image_features[i])", len(forged_image_features[i]))
        print("im_features==", im_features.shape)
        for i in range(len(genuine_image_features[i]) + len(forged_image_features[i])):
            print("i==", i)
            words, distance = vq(des_list[i][1], voc)
            print("des_list[i][1]===", des_list[i][1].shape)
            # print("distance=========",distance)
            for w in words:
                im_features[i][w] += 1

            for j in range(4):
                im_features[i][k + j] = im_contour_features[i][j]
                #print("im_features[i][k + j]", im_features[i][k + j])

        # Scaling the words
        stdSlr = StandardScaler().fit(im_features)
        print("stdSlr", stdSlr)
        im_features = stdSlr.transform(im_features)
        print("im_features===", im_features.shape)

        # train_genuine_features, test_genuine_features = im_features[0:3], im_features[3:5]
        test_genuine_features = im_features[1:5]
        test_forged_features = im_features[0:10]

        # train_forged_features, test_forged_features = im_features[5:8], im_features[8:10]
        print("test_genuine_features", test_genuine_features)
        print("test_forged_features", test_forged_features)
        print("length of the test_forged_features ", (test_forged_features.size))
        print("length of test_genuine_features", test_genuine_features.size)
        print("length of the test_forged_features ", test_forged_features.shape)
        print("length of test_genuine_features", test_genuine_features.shape)

        # Fitting model with training data
        clf = pickle.load(open("C:/Users/DELL/PycharmProjects/tiff to jpg/venv/clf66662.pkl", "rb"))
        # pickle.dump(clf,open("clf66662.pkl","wb"))

        genuine_res = clf.predict(test_genuine_features)
        print("genuine_res", genuine_res)

        # pickle.dump(test_genuine_features, open('test_genuine_features18-4-22.pkl', 'wb'))

        for res in genuine_res:
            if int(res) == 2:
                true_positive += 1
            else:
                false_negative += 1

        forged_res = clf.predict(test_forged_features)
        print("forged_res", forged_res)
        # pickle.dump(test_forged_features, open("test_forged_features18-4-22.pkl", "wb"))

        for res in forged_res:
            if int(res) == 1:
                true_negative += 1
            else:
                false_positive += 1


accuracy = float(
        ((true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive))) * 100
precision = float(((true_positive) / (true_positive + false_positive))) * 100
recall = float(((true_positive) / (true_positive + false_negative))) * 100
far = float(((false_positive) / (false_positive + true_negative))) * 100
frr = float(((false_negative) / (false_negative + true_positive))) * 100
f1_score = float(((2 * precision * recall) / (precision + recall))) * 100

print("True Positives: ", true_positive)
print("True Negatives: ", true_negative)
print("False Positives: ", false_positive)
print("False Negatives: ", false_negative)
print("Accuracy: ", round(accuracy, 2))
print("Precision: ", round(precision, 2))
print("Recall: ", round(recall, 2))
print("FAR: ", round(far, 2))
print("FRR: ", round(frr, 2))
print("F1 score: ", round(f1_score, 2))
arr = np.array([0])
forged_res=np.append(forged_res,arr)
diff=np.linalg.norm(genuine_res-forged_res)
print("The distance between 2 matrics are :",diff)
arr=np.array([0,0])
x = (genuine_res)
x =np.append(x,arr)
print(x)
y = (forged_res)

print(y)
# plt.title("distance graph")
# plt.xlabel("X axis")
# plt.ylabel("Y axis")
# plt.plot(x, y, color="red")
# plt.show()






