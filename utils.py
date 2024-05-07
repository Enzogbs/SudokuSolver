import cv2
import numpy as np
from keras.models import load_model

def biggest_contour(contours):
    max_area  = 0
    biggest = np.array([])
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder_vertices(vertices):
    vertices.reshape((4, 2))
    sum = vertices.sum(axis=2)
    diff = np.diff(vertices, axis=2)

    pt_A = vertices[np.argmin(sum)].reshape(2)
    pt_B = vertices[np.argmax(diff)].reshape(2)
    pt_C = vertices[np.argmax(sum)].reshape(2)
    pt_D = vertices[np.argmin(diff)].reshape(2)

    vertices = np.float32([pt_A, pt_B, pt_C, pt_D])
    return vertices


def output_dimensions(pt_A, pt_B, pt_C, pt_D):
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    return maxHeight, maxWidth


def SplitBoxes(img):
    boxes = []
    rows = np.vsplit(img, 9)
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (28, 28))
            boxes.append(box)
    boxes = np.array(boxes)
    return boxes


def process_warped_image(img):
    img, ret = cv2.threshold(img, 70, 255, cv2.THRESH_TOZERO)
    img, ret = cv2.threshold(ret, 70, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(ret)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def scale_and_center(img, ROI):
    if ROI:
        x, y, w, h = ROI
        ROI = img[y - 1:y + h + 1, x - 1:x + w + 1]
        ROI = cv2.resize(ROI, (12, 18))
        final = np.pad(ROI, ((5, 5), (8, 8)), "constant")
        return final
    else:
        ROI = np.zeros((28, 28))
        return ROI


def get_ROI(img, process=False):
    _, ret = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    if process:
        ret = cv2.GaussianBlur(ret, (3, 3), 1)
        kernel = np.ones((2, 2), np.uint8)
        ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)
        _, ret = cv2.threshold(ret, 45, 255, cv2.THRESH_BINARY)
        ret = cv2.morphologyEx(ret, cv2.MORPH_OPEN, kernel)
        ret = cv2.GaussianBlur(ret , (3, 3), 1)


    cnts = cv2.findContours(ret, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if (x < 3 or y < 3 or h < 3 or w < 3):
            continue
        return (x, y, w, h)

def extract(box):
    box[:3, :] = 0
    box[-3:,:] = 0
    box = box[4:26, 4:26]
    box = cv2.resize(box, (28, 28))

    ROI = get_ROI(box)
    box = scale_and_center(box, ROI)

    if np.sum(box) < 3000:
        return 0
    else:
        number = np.argmax(predict(box))
        return number

def get_answer_grid(answers):
    mask = np.zeros((450, 450, 3), dtype=np.float32)
    for line_index, line in enumerate(answers):
        for number_index, number in enumerate(line):
            if number != 0:
                number_mask = np.zeros((50, 50), dtype=np.float32)
                number_mask = cv2.cvtColor(number_mask, cv2.COLOR_GRAY2RGB)
                number_mask = cv2.putText(img=number_mask, text=str(number), org=(15, 35),
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                          color=(0, 255, 255), thickness=2)
                mask[line_index * 50: (line_index * 50) + 50, number_index * 50: (number_index * 50) + 50] += number_mask
    return mask


def predict(img):
    _, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    img = np.array(img)
    img = img.astype('float32')
    img = img.reshape(1, 28, 28, 1)
    img /= 255

    model = load_model("model.h5")
    classes = model.predict(img)
    return classes
