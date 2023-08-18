import os
import fitz
import cv2
import math
import numpy as np
import pandas as pd
import pytesseract as pt
import lib.pdf2plot.languages as languages

from lib.pdf2plot.document_page import DocumentPage

DPI = 300                       # image DPI
ZOOM = DPI/72                   # image zoom

class PDFToImage:
    def __init__(self, path = str, language = str, debug_mode = False, size_factor = float):
        self.__path = path
        self.__filename = os.path.basename(self.__path)[:-4]

        self.__pdf_doc = fitz.open(path)
        self.__out_path = os.path.join(os.getcwd(), 'out')
        self.__out_img_path = os.path.join(os.getcwd(), 'out', 'img')

        self.__out_plot_path = os.path.join(self.__out_img_path, 'plot')
        self.__out_matrix_path = os.path.join(self.__out_img_path, 'm_matrix')

        self.__magnify = fitz.Matrix(ZOOM, ZOOM)

        self.__out_img_pages = []
        self.__doc_pages = []

        self.__lang_dict = languages.LANGUAGE_DICT
        self.__lang = language

        self.__debug_mode = debug_mode

        self.__size_factor = size_factor if size_factor != 0.0 else 1.0

    @property
    def filename(self):
        return self.__filename

    @property
    def out_path(self):
        return self.__out_path
    
    @property
    def out_img_path(self):
        return self.__out_img_path
    
    @property
    def out_plot_path(self):
        return self.__out_plot_path
    
    @property
    def out_matrix_path(self):
        return self.__out_matrix_path
    
    @property
    def doc_pages(self):
        return self.__doc_pages
    
    @property
    def lang_dict(self):
        return self.__lang_dict
    
    @property
    def lang(self):
        return self.__lang

    def __page_to_img(self):
        for page in range(self.__pdf_doc.page_count):
            text = self.__pdf_doc[page].get_text()
            
            # getting the text-image
            pix_text = self.__pdf_doc[page].get_pixmap(matrix=self.__magnify)
            pix_text.set_dpi(DPI, DPI)
            im_text = np.frombuffer(pix_text.samples, dtype=np.uint8).reshape(pix_text.h, pix_text.w, pix_text.n)
            im_text = np.ascontiguousarray(im_text[..., [2, 1, 0]])          # rgb to bgr

            # getting the textless-image
            # it is formed only of not-colored vector-type shapes
            paths = self.__pdf_doc[page].get_drawings()
            page_textless = self.__pdf_doc.new_page(width=self.__pdf_doc[page].rect.width, height=self.__pdf_doc[page].rect.height)
            shape = page_textless.new_shape()

            for path in paths:
                # draw each entry of the 'items' list
                for item in path["items"]:                                              # these are the draw commands
                    if item[0] == "l":                                                  # line
                        shape.draw_line(item[1], item[2])
                    elif item[0] == "re":                                               # rectangle
                        shape.draw_rect(item[1])
                    elif item[0] == "qu":                                               # quad
                        shape.draw_quad(item[1])
                    elif item[0] == "c":                                                # curve
                        shape.draw_bezier(item[1], item[2], item[3], item[4])
                    else:
                        raise ValueError("unhandled drawing", item)

                shape.finish()

            shape.commit()

            pix_textless = page_textless.get_pixmap(matrix=self.__magnify)
            pix_textless.set_dpi(DPI, DPI)
            im_textless = np.frombuffer(pix_textless.samples, dtype=np.uint8).reshape(pix_textless.h, pix_textless.w, pix_textless.n)
            im_textless = np.ascontiguousarray(im_textless[..., [2, 1, 0]])

            self.__out_img_pages.append((im_text, im_textless, self.__filename, page))
            doc_page = DocumentPage(self.filename, page, im_text, im_textless, text, 300)         # 300 pixels is a test
            self.doc_pages.append(doc_page)

    # calculate y in Hough transform
    def __calc_y(self, x, rho, theta):
        if theta == 0:
            return rho
        else:
            return (-math.cos(theta) / math.sin(theta)) * x + (rho / math.sin(theta))
    
    # calculates (x,y) in Hough transform 
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

    # crops the image-format page using the Hough transform
    def __img_to_plot(self):
        for doc_page in self.doc_pages:
            print('Processing {file}.pdf page {pg}...'.format(file = doc_page.filename, pg = doc_page.page_number))

            page_copy = doc_page.vector_page.copy()
            page_gray = cv2.cvtColor(page_copy, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(page_gray, 180, 255, cv2.THRESH_BINARY)[1]
            res = pt.image_to_data(thresh, lang=self.lang, output_type = pt.Output.DICT)
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

            resize = (int(doc_page.vector_page.shape[1]/self.__size_factor), int(doc_page.vector_page.shape[0]/self.__size_factor))

            if self.__debug_mode:
                tmp_res = cv2.resize(page_copy, resize)
                cv2.imshow('Finding materiality matrix', tmp_res)
                cv2.waitKey(1500)

            image_gray = cv2.cvtColor(page_copy, cv2.COLOR_BGR2GRAY)
            _, work_image = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)

            edges = cv2.Canny(work_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, math.pi/180, 700)

            rows = []
            columns = []

            if self.__debug_mode:
                tmp = doc_page.raster_page.copy()

            height, width, _ = doc_page.raster_page.shape

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
                    cv2.waitKey(1500)

                tups = [(r, c) for r in rows for c in columns]

                if self.__debug_mode:
                    for t in tups:
                        cv2.circle(tmp, t, 6, (255, 0, 0), 3)
                        tmp_res = cv2.resize(tmp, resize)

                    cv2.imshow('Finding materiality matrix', tmp_res)
                    cv2.waitKey(1500)

                rows.sort(); columns.sort(reverse=True)

                try:
                    # intersection delimiter points
                    i_bottom_left = (rows[0], columns[0])                                                               # first x coord, first y coord
                    i_bottom_right = (rows[-1], columns[0])                                                             # least x coord, first y coord
                    i_top_left = (rows[0], columns[-1])                                                                 # first x coord, least y coord
                    i_top_right = (rows[-1], columns[-1])                                                               # least x coord, first y coord

                    if self.__debug_mode:
                        cv2.circle(tmp, i_bottom_left, 6, (0, 255, 0), 3)
                        cv2.circle(tmp, i_bottom_right, 6, (0, 255, 0), 3)
                        cv2.circle(tmp, i_top_left, 6, (0, 255, 0), 3)
                        cv2.circle(tmp, i_top_right, 6, (0, 255, 0), 3)

                        tmp_res = cv2.resize(tmp, resize)
                        cv2.imshow('Finding materiality matrix', tmp_res)
                        cv2.waitKey(1500)

                    upper_offset = 200                              # vertical offset
                    lower_offset = 350

                    # final cropped area delimiters
                    top_left = (0, i_top_left[1] - upper_offset) if (i_top_left[1] - upper_offset > 0) else (0, 0)
                    bottom_right = (width, i_bottom_right[1] + lower_offset) if (i_bottom_right[1] + lower_offset < height) else (width, height)

                    mat_matrix = doc_page.raster_page[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]                      # materiality matrix region
                    
                    fn_out = doc_page.filename + '_' + str(doc_page.page_number) + '.png'                                                      # output filename
                    
                    tmp = cv2.resize(mat_matrix, (int(mat_matrix.shape[1]/self.__size_factor), int(mat_matrix.shape[0]/self.__size_factor)))
                    cv2.imshow('Finding materiality matrix', tmp)

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # asking user if the cropped region is ok
                    while True:
                        choice = input('\nIs this crop ok? [Y/n] ')
                        # automatic crop is ok
                        if choice.lower()[0] == 'y':
                            if not os.path.isdir(self.out_path):
                                os.mkdir(self.out_path)

                            if not os.path.isdir(self.out_img_path):
                                os.mkdir(self.out_img_path)
                            
                            # if materiality matrix is in the page
                            # save in the relative folder
                            if self.lang_dict[self.lang] in doc_page.text.lower():
                                if not os.path.isdir(self.out_matrix_path):
                                    os.mkdir(self.out_matrix_path)
                                cv2.imwrite(os.path.join(self.out_matrix_path, fn_out), mat_matrix)
                                print(fn_out, 'was wrote in', self.out_matrix_path)
                            # else save into another directory
                            else:
                                if not os.path.isdir(self.out_plot_path):
                                    os.mkdir(self.out_matrix_path)
                                cv2.imwrite(os.path.join(self.out_plot_path, fn_out), mat_matrix)
                                print(fn_out, 'was wrote in', self.out_plot_path)
                            break
                        # user cropping
                        elif choice.lower()[0] == 'n':
                            resized_page = cv2.resize(doc_page.raster_page, resize)
                            roi = cv2.selectROI('Select the region of interest', resized_page)

                            if roi[0] != 0 and roi[1] != 0 and roi[2] != 0 and roi[3] != 0:
                                mat_matrix = doc_page.raster_page[int(roi[1]*self.__size_factor):int(roi[1]*self.__size_factor) + int(roi[3]*self.__size_factor),\
                                                                int(roi[0]*self.__size_factor):int(roi[0]*self.__size_factor) + int(roi[2]*self.__size_factor)]
                                resize_mat_matrix = (int(mat_matrix.shape[1]/self.__size_factor), int(mat_matrix.shape[0]/self.__size_factor))
                                tmp = cv2.resize(mat_matrix, resize_mat_matrix)
                                cv2.imshow('Selected materiality matrix', tmp)
                                cv2.waitKey(0)

                                if not os.path.isdir(self.out_path):
                                    os.mkdir(self.out_path)

                                if not os.path.isdir(self.out_img_path):
                                    os.mkdir(self.out_img_path)

                                # if materiality matrix is in the page
                                # save in the relative folder
                                if self.lang_dict[self.lang] in doc_page.text.lower():
                                    if not os.path.isdir(self.out_matrix_path):
                                        os.mkdir(self.out_matrix_path)
                                    cv2.imwrite(os.path.join(self.out_matrix_path, fn_out), mat_matrix)
                                    print(fn_out, 'was wrote in', self.out_matrix_path)
                                # else save into another directory
                                else:
                                    if not os.path.isdir(self.out_plot_path):
                                        os.mkdir(self.out_matrix_path)
                                    cv2.imwrite(os.path.join(self.out_plot_path, fn_out), mat_matrix)
                                    print(fn_out, 'was wrote in', self.out_plot_path)
                                break
                            else:
                                break
                        # invalid choice
                        else:
                            print('Invalid choice!')
                            continue

                except:
                    print('Not-legit intersection found, skipping to next page...')
                    continue
            else:
                continue
    
    def run(self):
        self.__page_to_img()
        self.__img_to_plot()