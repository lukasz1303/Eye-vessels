from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from skimage import filters, exposure
from tkinter import *
import threading
from keras_unet.utils import reconstruct_from_patches
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from keras.metrics import Precision, SensitivityAtSpecificity, SpecificityAtSensitivity
import random
import pickle


def findVessels(img, test, mask_img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    k = np.ones((4, 4))
    mask_img = cv2.erode(mask_img, k)
    size = gray.shape
    scale = size[0] / 600
    gray = cv2.bilateralFilter(gray, 5, 10, 10)

    median = filters.median(gray)
    clahe = exposure.equalize_adapthist(median)
    frangi = filters.frangi(clahe, sigmas=range(1, 10, 1))
    mask = np.where(frangi >= 0.75e-6)
    frangi[mask] = 1
    k = np.ones((7, 7))
    frangi = cv2.dilate(frangi, k)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    result = cv2.morphologyEx(frangi, cv2.MORPH_OPEN, kern)
    result = cv2.erode(result, k)
    result[mask_img == 0] = 0

    contours = cv2.findContours(result.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[-2]
    cnts = []
    for i in contours:
        if cv2.contourArea(i) < 350:
            cnts.append(i)
    cv2.drawContours(result, cnts, contourIdx=-1, color=(0, 0, 0), thickness=-1)
    cv2.imwrite("result2.jpg", result * 255)
    threading.Thread(target=display_results, args=(img, test, result, size, scale)).start()

    return result


def knn(img, test):
    # get saved classifier
    classifier = pickle.load(open('knnpickle_file_koniec', 'rb'))
    # create new classifier
    # classifier = train_knn()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    size = gray.shape
    scale = size[0] / 600
    gray = cv2.bilateralFilter(gray, 5, 10, 10)

    median = filters.median(gray)
    gray = exposure.equalize_adapthist(median)

    org = []
    img_parts = []

    for i in range(0, len(img) - 1, 5):
        for j in range(0, len(img[0]) - 1, 5):
            org.append(gray[i:i + 5, j:j + 5])
            img_parts.append((img[i:i + 5, j:j + 5]))

    parameters = []

    for i, j in zip(org, img_parts):
        moments = cv2.moments(i)
        hu = cv2.HuMoments(moments)
        vals = [*moments.values()][0:3]
        vals.append(hu[0][0])
        vals.append(hu[1][0])
        vals.append(hu[2][0])
        vals.append(np.var(i))
        vals.append(np.mean(j[:, :, 0]))
        vals.append(np.mean(j[:, :, 1]))
        vals.append(np.mean(j[:, :, 2]))
        vals.append(np.var(j[:, :, 0]))
        vals.append(np.var(j[:, :, 1]))
        vals.append(np.var(j[:, :, 2]))
        parameters.append(vals)

    del org
    del img_parts
    scaler = StandardScaler()
    scaler.fit(parameters)
    parameters = scaler.transform(parameters)
    pred = classifier.predict(parameters)

    result = np.zeros((size[0], size[1]))
    width = int((len(img[0]) - 1) / 5) + 1

    for i in range(0, int((len(img) - 1) / 5)):
        for j in range(0, width):
            if pred[i * width + j] == 1:
                result[i * 5:i * 5 + 5, j * 5:j * 5 + 5] = np.ones((5, 5))

    threading.Thread(target=display_results, args=(img, test, result, size, scale)).start()

    return result


def train_knn():
    files = [f for f in listdir("train") if isfile(join("train", f))]
    images = []
    for img in files:
        fullname = "train/" + img
        images.append(cv2.imread(fullname))

    files2 = [f for f in listdir("test_train") if isfile(join("test_train", f))]
    tests = []
    for img in files2:
        fullname = "test_train/" + img
        tests.append(cv2.imread(fullname))

    decisions = []
    parameters = []

    for img, test in zip(images, tests):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        test2 = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 10, 10)
        median = filters.median(gray)
        gray = exposure.equalize_adapthist(median)

        randoms_width = random.sample(range(0, len(img) - 5, 5), 250)
        randoms_height = random.sample(range(0, len(img[0]) - 5, 5), 250)

        ok = 0
        not_ok = 0

        for i in randoms_width:
            for j in randoms_height:
                decision = int(test2[i + 2][j + 2] / 255)
                if decision == 1:
                    ok += 1
                else:
                    if not_ok > 4 * ok:
                        continue
                    else:
                        not_ok += 1
                decisions.append(decision)
                org = gray[i:i + 5, j:j + 5]
                img_part = (img[i:i + 5, j:j + 5])

                moments = cv2.moments(org)
                hu = cv2.HuMoments(moments)
                vals = [*moments.values()][0:3]
                vals.append(hu[0][0])
                vals.append(hu[1][0])
                vals.append(hu[2][0])
                vals.append(np.var(i))
                vals.append(np.mean(img_part[:, :, 0]))
                vals.append(np.mean(img_part[:, :, 1]))
                vals.append(np.mean(img_part[:, :, 2]))
                vals.append(np.var(img_part[:, :, 0]))
                vals.append(np.var(img_part[:, :, 1]))
                vals.append(np.var(img_part[:, :, 2]))
                parameters.append(vals)

    del images
    del tests

    X_train, X_test, y_train, y_test = train_test_split(parameters, decisions, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=10)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f1_score(y_test, y_pred))

    knnPickle = open('knnpickle_file', 'wb')
    pickle.dump(classifier, knnPickle)

    return classifier


def dnn(img, test):
    patch_size = [48, 48]
    # Wczytanie danych, pobranie wycinków oraz podział na zbiór trenujący oraz testowy
    x_train_data, y_train_data = getTrainData()
    x_train_data = (np.array(x_train_data))
    y_train_data = (np.array(y_train_data))

    x_data, y_data = getRandomPatches(x_train_data, y_train_data, patch_size)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

    model = getUnet((patch_size[0], patch_size[1], 1))

    # Trenowanie modelu
    # model.fit(X_train, y_train, epochs=20, batch_size=32,verbose=1, shuffle=True, validation_split=0.1)
    # model.save_weights('last_weights.h5', overwrite=True)

    model.load_weights('last_weights.h5')
    # Test nauczonej sieci
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test precision:', score[2])
    print('Test sensitivity:', score[3])
    print('Test specifity:', score[4])
    g_mean = (score[3] * score[4]) ** 0.5
    print("g_mean:", g_mean)

    gray = preProcess(img)
    size = gray.shape
    scale = size[0] / 600
    base_height = gray.shape[0]
    base_width = gray.shape[1]
    x_predict, new_shape = getAllPatches(gray, patch_size)
    x_predict = np.reshape(x_predict, (x_predict.shape[0], x_predict.shape[1], x_predict.shape[2], 1))

    predictions = model.predict(x_predict, batch_size=32, verbose=1)

    x_reconstructed = reconstruct_from_patches(img_arr=predictions, org_img_size=(new_shape[0], new_shape[1]))
    x_reconstructed = np.reshape(x_reconstructed, (x_reconstructed.shape[1], x_reconstructed.shape[2]))

    result = x_reconstructed[0:base_height, 0:base_width]

    ret, thresh = cv2.threshold(result * 255, 60, 255, cv2.THRESH_BINARY)
    final_result = thresh / 255

    threading.Thread(target=display_results, args=(img, test, final_result, size, scale)).start()

    return final_result


def getUnet(shape):
    inputs = Input(shape=shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv5)

    conv6 = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=conv6)

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', Precision(), SensitivityAtSpecificity(0.5), SpecificityAtSensitivity(0.5)])

    return model


def getTrainData():
    files = [f for f in listdir("train") if isfile(join("train", f))]
    images = []
    for img in files:
        fullname = "train/" + img
        im = cv2.imread(fullname)
        images.append(preProcess(im))

    files2 = [f for f in listdir("test_train") if isfile(join("test_train", f))]
    tests = []
    for img in files2:
        fullname = "test_train/" + img
        data = cv2.imread(fullname)
        gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY) / 255
        tests.append(gray)

    return images, tests


def getRandomPatches(images, tests, p_size):
    patches_per_image = 500
    img_patches = []
    test_patches = []
    for i in range(len(images)):
        img_height = images[i].shape[0]
        img_width = images[i].shape[1]
        for j in range(patches_per_image):
            x = random.randint(0, img_height - p_size[0])
            y = random.randint(0, img_width - p_size[1])
            img_patch = images[i][x:(x + p_size[0]), y:(y + p_size[1])]
            test_patch = tests[i][x:(x + p_size[0]), y:(y + p_size[1])]
            img_patches.append(img_patch)
            test_patches.append(test_patch)

    return img_patches, test_patches


def getAllPatches(img, p_size):
    img_height = img.shape[0]
    img_width = img.shape[1]
    p_height = p_size[0]
    p_width = p_size[1]
    h_left = img_height % p_height
    w_left = img_width % p_width
    h_diff = 0
    w_diff = 0
    if h_left != 0:
        h_diff = p_height - h_left
    if w_left != 0:
        w_diff = p_width - w_left
    new_img_height = img.shape[0] + h_diff
    new_img_width = img.shape[1] + w_diff
    new_img = np.zeros((new_img_height, new_img_width))
    new_img[0:img.shape[0], 0:img.shape[1]] = img

    N_patches_h = int(new_img_height / p_height)
    N_patches_w = int(new_img_width / p_width)
    N_patches_tot = (N_patches_h * N_patches_w)
    patches = np.empty((N_patches_tot, p_height, p_width, 1))

    iter_tot = 0
    for h in range(N_patches_h):
        for w in range(N_patches_w):
            patch = new_img[h * p_height:(h * p_height) + p_height, w * p_width:(w * p_width) + p_width]
            patch = np.resize(patch, (p_height, p_width, 1))
            patches[iter_tot] = patch
            iter_tot += 1

    return patches, [new_img_height, new_img_width]


def preProcess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 10, 10)
    median = filters.median(gray)
    gray = exposure.equalize_adapthist(median)

    return gray


def run(img, test, mask_img, method):
    cv2.destroyAllWindows()
    processing_label = Label(app, text="Przetwarzanie obrazu ... ", font=('serif', 9))
    processing_label.place(x=100, y=130)

    accuracy_value = Label(app, text="                  ", font=('serif', 9))
    accuracy_value.place(x=200, y=150)
    sensitivity_value = Label(app, text="                  ", font=('serif', 9))
    sensitivity_value.place(x=200, y=170)
    specificity_value = Label(app, text="                  ", font=('serif', 9))
    specificity_value.place(x=200, y=190)
    precision_value = Label(app, text="                  ", font=('serif', 9))
    precision_value.place(x=200, y=210)
    g_mean_value = Label(app, text="                  ", font=('serif', 9))
    g_mean_value.place(x=200, y=230)
    f_measure_value = Label(app, text="                  ", font=('serif', 9))
    f_measure_value.place(x=200, y=250)
    TP_value = Label(app, text="                  ", font=('serif', 9))
    TP_value.place(x=200, y=270)
    TN_value = Label(app, text="                  ", font=('serif', 9))
    TN_value.place(x=200, y=290)
    FP_value = Label(app, text="                  ", font=('serif', 9))
    FP_value.place(x=200, y=310)
    FN_value = Label(app, text="                  ", font=('serif', 9))
    FN_value.place(x=200, y=330)

    if method == "image processing":
        result = findVessels(img, test, mask_img)
    elif method == "knn":
        result = knn(img, test)
    else:
        result = dnn(img, test)  # TODO NEURAL NETWORK

    processing_label.destroy()
    processing_label = Label(app, text="Obliczanie metryk ... ", font=('serif', 9))
    processing_label.place(x=100, y=130)

    TP, TN, FP, FN, Accuracy, Sensitivity, Specificity, Precision, G_Mean, F_Measure = calculate_metrics(result, test)

    accuracy_value = Label(app, text=Accuracy.__round__(5), font=('serif', 9))
    accuracy_value.place(x=200, y=150)
    sensitivity_value = Label(app, text=Sensitivity.__round__(5), font=('serif', 9))
    sensitivity_value.place(x=200, y=170)
    specificity_value = Label(app, text=Specificity.__round__(5), font=('serif', 9))
    specificity_value.place(x=200, y=190)
    precision_value = Label(app, text=Precision.__round__(5), font=('serif', 9))
    precision_value.place(x=200, y=210)
    g_mean_value = Label(app, text=G_Mean.__round__(5), font=('serif', 9))
    g_mean_value.place(x=200, y=230)
    f_measure_value = Label(app, text=F_Measure.__round__(5), font=('serif', 9))
    f_measure_value.place(x=200, y=250)
    TP_value = Label(app, text=TP.__round__(5), font=('serif', 9))
    TP_value.place(x=200, y=270)
    TN_value = Label(app, text=TN.__round__(5), font=('serif', 9))
    TN_value.place(x=200, y=290)
    FP_value = Label(app, text=FP.__round__(5), font=('serif', 9))
    FP_value.place(x=200, y=310)
    FN_value = Label(app, text=FN.__round__(5), font=('serif', 9))
    FN_value.place(x=200, y=330)

    processing_label.destroy()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    accuracy_value.destroy()
    sensitivity_value.destroy()
    specificity_value.destroy()
    precision_value.destroy()
    g_mean_value.destroy()
    f_measure_value.destroy()


def display_results(img, test, result, size, scale):
    cv2.imshow("Oryginal", cv2.resize(img, (int(size[1] / scale), int(size[0] / scale))))
    cv2.imshow("Manual", cv2.resize(test, (int(size[1] / scale), int(size[0] / scale))))
    cv2.waitKey(1)
    cv2.imshow("Result", cv2.resize(result, (int(size[1] / scale), int(size[0] / scale))))
    cv2.imwrite("result1d.jpg", result * 255)
    cv2.waitKey(1)

    truth_table_mask_image = np.zeros((img.shape[0], img.shape[1], 3))
    test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)

    for i in range(len(result)):
        for j in range(len(result[0])):
            if int(result[i][j]) == 1 and int(test[i][j] / 255) == 1:
                truth_table_mask_image[i][j] = [255, 255, 255]
            elif int(result[i][j]) == 0 and int(test[i][j] / 255) == 1:
                truth_table_mask_image[i][j] = [0, 0, 255]
            elif int(result[i][j]) == 1 and int(test[i][j] / 255) == 0:
                truth_table_mask_image[i][j] = [0, 255, 0]
            else:
                truth_table_mask_image[i][j] = [0, 0, 0]

    truth_table_mask_image[len(truth_table_mask_image) - 160: len(truth_table_mask_image) - 20,
    len(truth_table_mask_image[0]) - 830: len(truth_table_mask_image[0]) - 10] = np.ones((140, 820, 3)) * 255

    color = np.ones((60, 60, 3)) * 255
    color[0:5, :, :] -= 255
    color[:, 0:5, :] -= 255
    color[len(color) - 5:len(color), :, :] -= 255
    color[:, len(color[0]) - 5:len(color[0]), :] -= 255
    truth_table_mask_image[len(truth_table_mask_image) - 120: len(truth_table_mask_image) - 60,
    len(truth_table_mask_image[0]) - 700: len(truth_table_mask_image[0]) - 640] = color
    cv2.putText(truth_table_mask_image, "TP", (len(truth_table_mask_image[0]) - 800, len(truth_table_mask_image) - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0, 255), 3)

    color = np.zeros((60, 60, 3))
    color[0:5, :, :] += 255
    color[:, 0:5, :] += 255
    truth_table_mask_image[len(truth_table_mask_image) - 120: len(truth_table_mask_image) - 60,
    len(truth_table_mask_image[0]) - 500: len(truth_table_mask_image[0]) - 440] = color
    cv2.putText(truth_table_mask_image, "TN", (len(truth_table_mask_image[0]) - 600, len(truth_table_mask_image) - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0, 255), 3)

    color = np.zeros((60, 60, 3))
    color[:, :, 1] += 255
    color[0:5, :, :] -= 255
    color[:, 0:5, :] -= 255
    color[len(color) - 5:len(color), :, :] -= 255
    color[:, len(color[0]) - 5:len(color[0]), :] -= 255
    truth_table_mask_image[len(truth_table_mask_image) - 120: len(truth_table_mask_image) - 60,
    len(truth_table_mask_image[0]) - 300: len(truth_table_mask_image[0]) - 240] = color
    cv2.putText(truth_table_mask_image, "FP", (len(truth_table_mask_image[0]) - 400, len(truth_table_mask_image) - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0, 255), 3)

    color = np.zeros((60, 60, 3))
    color[:, :, 2] += 255
    color[0:5, :, :] -= 255
    color[:, 0:5, :] -= 255
    color[len(color) - 5:len(color), :, :] -= 255
    color[:, len(color[0]) - 5:len(color[0]), :] -= 255
    truth_table_mask_image[len(truth_table_mask_image) - 120: len(truth_table_mask_image) - 60,
    len(truth_table_mask_image[0]) - 100: len(truth_table_mask_image[0]) - 40] = color
    cv2.putText(truth_table_mask_image, "FN", (len(truth_table_mask_image[0]) - 200, len(truth_table_mask_image) - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0, 255), 3)

    cv2.imwrite("truth_table1d.jpg", truth_table_mask_image)
    cv2.imshow("truth_table", cv2.resize(truth_table_mask_image, (int(size[1] / scale), int(size[0] / scale))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_metrics(result, original):
    original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) / 255
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(result)):
        for j in range(len(result[0])):
            if int(result[i][j]) == 1 and int(original[i][j]) == 1:
                TP += 1
            if int(result[i][j]) == 0 and int(original[i][j]) == 0:
                TN += 1
            if int(result[i][j]) == 1 and int(original[i][j]) == 0:
                FP += 1
            if int(result[i][j]) == 0 and int(original[i][j]) == 1:
                FN += 1

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Precision = TP / (TP + FP)
    G_Mean = (Sensitivity * Specificity) ** 0.5
    F_Measure = (2 * Precision * Sensitivity) / (Precision + Sensitivity)

    return TP, TN, FP, FN, Accuracy, Sensitivity, Specificity, Precision, G_Mean, F_Measure


if __name__ == '__main__':
    app = Tk()
    app.resizable(False, False)
    app.geometry("400x380")
    app.title("Dno oka")
    choose = StringVar(app)
    choose2 = StringVar(app)
    files = [f for f in listdir("img") if isfile(join("img", f))]
    files2 = [f for f in listdir("test") if isfile(join("test", f))]
    files3 = [f for f in listdir("masks") if isfile(join("masks", f))]
    images = []
    tests = []
    masks = []

    for img in files:
        fullname = "img/" + img
        images.append(cv2.imread(fullname))
    for img in files2:
        fullname2 = "test/" + img
        tests.append(cv2.imread(fullname2))
    for img in files3:
        fullname3 = "masks/" + img
        masks.append(cv2.imread(fullname3))
    choose.set(files[0])
    images_list = OptionMenu(app, choose, *files)
    images_list.config(width=27)
    images_list.place(x=100, y=60)

    methods = ["image processing", "knn", "dnn"]
    choose2.set(methods[0])
    images_list = OptionMenu(app, choose2, *methods)
    images_list.config(width=27)
    images_list.place(x=100, y=20)

    accuracy_label = Label(app, text="Accuracy: ", font=('serif', 9))
    accuracy_label.place(x=100, y=150)
    sensitivity_label = Label(app, text="Sensitivity: ", font=('serif', 9))
    sensitivity_label.place(x=100, y=170)
    specificity_label = Label(app, text="Specificity: ", font=('serif', 9))
    specificity_label.place(x=100, y=190)
    precision_label = Label(app, text="Precision: ", font=('serif', 9))
    precision_label.place(x=100, y=210)
    g_mean_label = Label(app, text="G_Mean: ", font=('serif', 9))
    g_mean_label.place(x=100, y=230)
    f_measure_label = Label(app, text="F_Measure: ", font=('serif', 9))
    f_measure_label.place(x=100, y=250)
    TP_label = Label(app, text="TP: ", font=('serif', 9))
    TP_label.place(x=100, y=270)
    TN_label = Label(app, text="TN: ", font=('serif', 9))
    TN_label.place(x=100, y=290)
    FP_label = Label(app, text="FP: ", font=('serif', 9))
    FP_label.place(x=100, y=310)
    FN_label = Label(app, text="FN: ", font=('serif', 9))
    FN_label.place(x=100, y=330)

    start = Button(app, command=lambda: threading.Thread(target=lambda: run(
        images[files.index(choose.get())], tests[files.index(choose.get())], masks[files.index(choose.get())],
        choose2.get())).start(), text="Process", height=1,
                   width=28).place(x=100, y=100)
    app.mainloop()
