import os
import fitz
import cv2
import math
import numpy as np

pdf_name = input('Enter pdf name: ')

pdf_path = os.path.join(os.getcwd(), pdf_name)

pdf_doc = fitz.open(pdf_path)
print(pdf_doc)

DPI = 300
ZOOM = DPI / 72                                         # zoom factor, standard dpi is 72
magnify = fitz.Matrix(ZOOM, ZOOM)                       # takes care of zooming

out_img_pages = []

def calc_y(x, rho, theta):
    if theta == 0:
        return rho
    else:
        return (-math.cos(theta) / math.sin(theta)) * x + (rho / math.sin(theta))

def polar_to_xy(rho, theta, width):
    if theta == 0:
        x1 = rho
        x2 = rho
        y1 = 0
        y2 = width
    else:
        x1 = 0
        x2 = width
        y1 = int(calc_y(0, rho, theta))
        y2 = int(calc_y(width, rho, theta))

    return (int(x1), int(y1)), (int(x2), int(y2))

for page in pdf_doc:
    text = page.get_text()

    if 'matrice di materialità' in text.lower():
        pix = page.get_pixmap(matrix=magnify)
        pix.set_dpi(DPI, DPI)
        # img_name = pdf_name[:-4] + '_p' + str(page.number) + '.png'
        # img_path = os.path.join(os.getcwd(), img_name)
        # pix.save(img_path)
        im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        im = np.ascontiguousarray(im[..., [2, 1, 0]])          # rgb to bgr
        out_img_pages.append(im)

for page in out_img_pages:
    resize = (int(page.shape[1]/3.5), int(page.shape[0]/3.5))

    image_gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    _, work_image = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(work_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, math.pi/180, 475)

    rows = []
    columns = []

    print(len(lines))

    tmp = page.copy()

    height, width, _ = page.shape

    for rho, theta, in lines[:,0,]:
        (x1, y1), (x2, y2) = polar_to_xy(rho, theta, width)
        cv2.line(tmp, (x1, y1), (x2, y2), (0, 0, 255), 4)

    tmp_res = cv2.resize(tmp, resize)

    cv2.imshow('Finding image limits', tmp_res)
    cv2.waitKey(0)
