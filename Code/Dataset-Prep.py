import os
import shutil
import cv2
import numpy as np
import hashlib
import time
import matplotlib.pyplot as plt

def show_image(image, figsize=(7,7), cmap=None):
    cmap = cmap if len(image.shape)==3 else 'gray'
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.show()

def AdaptiveThresh(gray):
    blur = cv2.medianBlur(gray, 5)
    adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV
    return cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)

def get_rect(pts):
    xmin = pts[:,0,1].min()
    ymin = pts[:,0,0].min()
    xmax = pts[:,0,1].max()
    ymax = pts[:,0,0].max()
    return (ymin,xmin), (ymax,xmax)

def save_image(images, name_code):
    classes = []

    if name_code == '1':
        classes = [ '0', '1', '2', '3', '4', '5', '6',
                    '7', '8', '9', 'a', 'i', 'u', 'e',
                    'an', 'in', 'un', 'b', 'b', 'b', 'b',
                    't', 't', 't', 't', 't', 't', 'ṡ',
                    'ṡ', 'ṡ', 'ṡ', 'j', 'j', 'j', 'j',
                    'c', 'c', 'c', 'c', 'ḥ', 'ḥ', 'ḥ',
                    'ḥ', 'kh', 'kh', 'kh', 'kh', 'd', 'd',
                    'dz', 'dz', 'dh', 'dh', 'r', 'r', 'z',
                    'z', 's', 's', 's', 's', 'sy', 'sy',
                ]
        
    elif name_code == '2':
        classes = [ 'sy', 'sy', 'ṣ', 'ṣ', 'ṣ', 'ṣ', 'ḍ',
                    'ḍ', 'ḍ', 'ḍ', 'ṭ', 'ṭ', 'ṭ', 'ṭ',
                    'th', 'th', 'th', 'th', 'ẓ', 'ẓ', 'ẓ',
                    'ẓ', 'gh', 'gh', 'gh', 'gh', 'ng', 'ng',
                    'ng', 'ng', 'f', 'f', 'f', 'f', 'p',
                    'p', 'p', 'p', 'q', 'q', 'q', 'q',
                    'k', 'k', 'k', 'k', 'g', 'g', 'g',
                    'g', 'l', 'l', 'l', 'l', 'l', 'm',
                    'm', 'm', 'm', 'n', 'n', 'n', 'n',
                ]
        
    for i in range(len(classes)):
            timestamp = str(time.time())
            md5_hash = hashlib.md5()
            md5_hash.update(timestamp.encode('utf-8'))
            md5_result = md5_hash.hexdigest()

            file_name = './Data' + '/' + classes[i] + '/' + md5_result + '.png'
            cv2.imencode('.png', images[i])[1].tofile(file_name)

            print(f'Generate | {file_name}')

if not os.path.exists('./Data'):
    os.makedirs('./Data')

for item in os.listdir('./Data'):
        item_path = os.path.join('./Data', item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'i', 'u', 'e',
    'an', 'in', 'un', 'b', 't', 'ṡ', 'j', 'c', 'ḥ', 'kh', 'd', 'dz', 'dh', 
    'r', 'z', 's', 'sy', 'ṣ', 'ḍ', 'ṭ', 'th', 'ẓ', 'gh', 'ng', 'f', 'p', 'q', 
    'k', 'g', 'l', 'm', 'n'
]

for i in classes:
    os.makedirs('./Data/' + i)

for root, dirs, files in os.walk('./Documents'):
    folder_name = os.path.basename(root)
    if folder_name.startswith('Hand-Writing-') and folder_name[13:].isdigit():
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_code = os.path.splitext(file)[0]

                image_path = os.path.join(root, file)
                image_name = image_path
                image_original = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
                image_gray = 255-cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)

                kernel = np.ones((3,3),np.uint8)
                d = 255-cv2.dilate(image_gray,kernel,iterations = 1)

                e = AdaptiveThresh(d)

                m = cv2.dilate(e,kernel,iterations = 1)
                m = cv2.medianBlur(m,11)
                m = cv2.dilate(m,kernel,iterations = 1)

                contours, hierarchy = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                total_area = np.prod(image_gray.shape)
                max_area = 0
                for cnt in contours:
                    perimeter = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
                    area = cv2.contourArea(approx)

                    if (len(approx) == 4 and cv2.isContourConvex(approx) and max_area<area<total_area):
                        max_area = cv2.contourArea(approx)
                        quad_polygon = approx

                img1 = image_original.copy()
                img2 = image_original.copy()

                cv2.polylines(img1,[quad_polygon],True,(0,255,0),10)
                tl, br = get_rect(quad_polygon)

                cv2.rectangle(img2, tl, br, (0,255,0), 10)
                cv2.polylines(img2, [quad_polygon], True, (0, 255, 0), 10)

                pts = quad_polygon.reshape(4, 2)
                rect = np.zeros((4, 2), dtype='float32')

                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                (tl, tr, br, bl) = rect

                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))

                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype='float32')

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), M, (maxWidth, maxHeight))

                image = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

                rows = 9
                cols = 7

                height, width, _ = image.shape

                tile_height = height // rows
                tile_width = width // cols

                fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

                images = []

                for i in range(rows):
                    for j in range(cols):
                        start_x = j * tile_width
                        start_y = i * tile_height
                        end_x = (j + 1) * tile_width
                        end_y = (i + 1) * tile_height

                        sub_image = image[start_y +120:end_y -15, start_x +60:end_x -15]
                        sub_image_gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
                        sub_image_binary = cv2.threshold(sub_image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        sub_image_binary_inv = 255 - sub_image_binary
                        final = cv2.resize(sub_image_binary_inv, (50, 50))

                        images.append(final)

                save_image(images, class_code)