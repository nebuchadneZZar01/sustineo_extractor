import os
import fitz
import cv2
import argparse
import math
import numpy as np
import pandas as pd
import pytesseract as pt

DPI = 300
ZOOM = DPI/72

class PDFToImage:
    def __init__(self, path, language, debug_mode = False):
        self.__path = path
        self.__filename = os.path.basename(self.__path)[:-4]

        self.__pdf_doc = fitz.open(path)
        self.__out_path = os.path.join(os.getcwd(), 'src', 'img')

        self.__magnify = fitz.Matrix(ZOOM, ZOOM)

        self.__out_img_pages = []

        self.__lang_dict = {'eng': 'materiality matrix', 'ita': 'matrice di materialità'}
        self.__lang = language

        self.__debug_mode = debug_mode

    def __page_to_img(self):
        for page in range(self.__pdf_doc.page_count):
            text = self.__pdf_doc[page].get_text()
            
            if self.__lang_dict[self.__lang] in text.lower():
                pix = self.__pdf_doc[page].get_pixmap(matrix=self.__magnify)
                pix.set_dpi(DPI, DPI)
                im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                im = np.ascontiguousarray(im[..., [2, 1, 0]])          # rgb to bgr
                self.__out_img_pages.append((im, self.__filename, page))

    def __calc_y(self, x, rho, theta):
        if theta == 0:
            return rho
        else:
            return (-math.cos(theta) / math.sin(theta)) * x + (rho / math.sin(theta))
    
    def __polar_to_xy(self, rho, theta, width):
        if theta == 0:
            x1 = rho
            x2 = rho
            y1 = 0
            y2 = width
        else:
            x1 = 0
            x2 = width
            y1 = int(self.__calc_y(0, rho, theta))
            y2 = int(self.__calc_y(width, rho, theta))

        return (int(x1), int(y1)), (int(x2), int(y2))

    def __img_to_plot(self):
        for page in self.__out_img_pages:
            print('Processing page {pg}...'.format(pg = page[2]))
            resize = (int(page[0].shape[1]/3.5), int(page[0].shape[0]/3.5))

            page_copy = page[0].copy()
            page_gray = cv2.cvtColor(page[0], cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(page_gray, 180, 255, cv2.THRESH_BINARY)[1]
            res = pt.image_to_data(thresh, lang=self.__lang, output_type = pt.Output.DICT)
            res = pd.DataFrame(res)
            res = res.loc[res['conf'] != -1]

            for i in range(0, len(res)):
                # extract the bounding box coordinates of the text region from
                # the current result
                x = res.iloc[i]['left']
                y = res.iloc[i]['top']
                w = res.iloc[i]['width']
                h = res.iloc[i]['height']
                # extract the OCR text itself along with the confidence of the
                # text localization
                text = res.iloc[i]['text']
                conf = int(res.iloc[i]['conf'])

                if conf > 65:
                    if len(text) > 1:
                        cv2.rectangle(page_copy, (x, y), (x + w, y + h), (255, 255, 255), -1)

            if self.__debug_mode:
                tmp_res = cv2.resize(page_copy, resize)
                cv2.imshow('Finding materiality matrix', tmp_res)
                cv2.waitKey(0)

            image_gray = cv2.cvtColor(page_copy, cv2.COLOR_BGR2GRAY)
            _, work_image = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)

            edges = cv2.Canny(work_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, math.pi/180, 550)

            rows = []
            columns = []

            if self.__debug_mode:
                tmp = page[0].copy()

            height, width, _ = page[0].shape

            if lines is not None:
                for rho, theta, in lines[:,0,]:
                    (x1, y1), (x2, y2) = self.__polar_to_xy(rho, theta, width)

                    if theta == 0:
                        rows.append(int(rho))
                    else:
                        columns.append(int(rho))
                    
                    if self.__debug_mode:
                        cv2.line(tmp, (x1, y1), (x2, y2), (0, 0, 255), 4)

                if self.__debug_mode:
                    tmp_res = cv2.resize(tmp, resize)
                    cv2.imshow('Finding materiality matrix', tmp_res)
                    cv2.waitKey(0)

                tups = [(r, c) for r in rows for c in columns]

                if self.__debug_mode:
                    for t in tups:
                        cv2.circle(tmp, t, 6, (255, 0, 0), 3)
                        tmp_res = cv2.resize(tmp, resize)

                    cv2.imshow('Finding materiality matrix', tmp_res)
                    cv2.waitKey(0)

                rows.sort(); columns.sort(reverse=True)

                try:
                    # print('rows:', rows)
                    # print('columns:', columns)
                    
                    l_bottom_left = (rows[0], columns[0])
                    l_bottom_right = (rows[2], columns[0])
                    l_top_left = (rows[0], columns[3])

                    # size of the rectangle's edges
                    l_w = l_bottom_right[0] - l_bottom_left[0]
                    l_h = l_bottom_left[1] - l_top_left[1]

                    if self.__debug_mode:
                        cv2.circle(tmp, l_bottom_left, 6, (0, 255, 0), 3)
                        cv2.circle(tmp, l_bottom_right, 6, (0, 255, 0), 3)
                        cv2.circle(tmp, l_top_left, 3, (0, 255, 0), 3)

                        tmp_res = cv2.resize(tmp, resize)
                        cv2.imshow('Finding materiality matrix', tmp_res)
                        cv2.waitKey(0)

                    b_h = 3 * l_h
                    b_w = 3 * l_w

                    # vertices delimiting the rectangle
                    b_bottom_left = l_bottom_left
                    # if the vertex calculated coordinates are beyond the image size
                    # then it is set exactly on the border of the image
                    b_top_left = (l_bottom_left[0], l_bottom_left[1] - b_h) if (l_bottom_left[1] - b_h) > 0 else (l_bottom_left[0], 0)
                    b_bottom_right = (l_bottom_left[0] + b_w, l_bottom_left[1]) if (l_bottom_left[0] + b_w < tmp.shape[0]) else (tmp.shape[0], l_bottom_left[1])

                    v_offset = 400                              # vertical offset
                    h_offset = 200                              # horizontal offset

                    bottom_right = (b_bottom_right[0] + h_offset, b_bottom_right[1] + v_offset)
                    top_left = (b_top_left[0] - h_offset, b_top_left[1])

                    w = top_left[0] - bottom_right[0] if (top_left[0] - bottom_right[0]) > 0 else -1 * (top_left[0] - bottom_right[0])
                    h = bottom_right[1] - top_left[0] if (bottom_right[1] - top_left[0]) > 0 else -1 * (bottom_right[1] - top_left[0])

                    mat_matrix = page[0][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    
                    fn_out = page[1] + '_' + str(page[2]) + '.png'
                    
                    if self.__debug_mode:
                        tmp = cv2.resize(mat_matrix, (int(mat_matrix.shape[1]/3), int(mat_matrix.shape[0]/3)))
                        cv2.imshow('Finding materiality matrix', tmp)
                        cv2.waitKey(0)

                    cv2.imwrite(os.path.join(self.__out_path, fn_out), mat_matrix)
                    print(fn_out, 'was wrote in', self.__out_path)
                except:
                    print('Not-legit intersection found, skipping to next page...')
                    continue
            else:
                continue
    
    def run(self):
        self.__page_to_img()
        self.__img_to_plot()

def main(pdf_path, language, debug):
    extr = PDFToImage(pdf_path, language, debug)
    extr.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sustineo_extractor',
                                description='This program extracts materiality matrices in raster-format from pdf files.\
                                            \nAuthor: nebuchadneZZar01 (Michele Ferro)\
                                            \nGitHub: https://github.com/nebuchadneZZar01/',\
                                            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('filename')

    parser.add_argument('-l', '--language', type=str, default='ita',\
                        help='language of the plot to extract (default="ita")')

    parser.add_argument('-d', '--debug-mode', action='store_true',\
                        help='activate the visualization of the various passes')

    args = parser.parse_args()

    if os.path.isfile(args.filename):
        main(args.filename, args.language, args.debug_mode)
    else:
        print('ERROR: File {fn} does not exist'.format(fn = args.filename))