import os
import fitz
import cv2
import argparse
import math
import numpy as np
import pandas as pd
import pytesseract as pt
import languages

DPI = 300                       # image DPI
ZOOM = DPI/72                   # image zoom

class PDFToImage:
    def __init__(self, path = str, language = str, debug_mode = False):
        self.__path = path
        self.__filename = os.path.basename(self.__path)[:-4]

        self.__pdf_doc = fitz.open(path)
        self.__out_path = os.path.join(os.getcwd(), 'src', 'img')

        self.__magnify = fitz.Matrix(ZOOM, ZOOM)

        self.__out_img_pages = []

        self.__lang_dict = languages.LANGUAGE_DICT
        self.__lang = language

        self.__debug_mode = debug_mode

    def __page_to_img(self):
        for page in range(self.__pdf_doc.page_count):
            text = self.__pdf_doc[page].get_text()
            
            if self.__lang_dict[self.__lang] in text.lower():
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
        for page in self.__out_img_pages:
            # page[0] -> original image
            # page[1] -> vectors image
            # page[2] -> filename
            # page[3] -> page number
            print('Processing {file}.pdf page {pg}...'.format(file = page[2], pg = page[3]))
            resize = (int(page[1].shape[1]/3.5), int(page[1].shape[0]/3.5))

            page_copy = page[1].copy()
            page_gray = cv2.cvtColor(page_copy, cv2.COLOR_BGR2GRAY)
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
            lines = cv2.HoughLines(edges, 1, math.pi/180, 700)

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

                    v_offset = 350                              # vertical offset

                    # final cropped area delimiters
                    top_left = (0, i_top_left[1] - v_offset) if (i_top_left[1] - v_offset > 0) else (0, 0)
                    bottom_right = (width, i_bottom_right[1] + v_offset) if (i_bottom_right[1] + v_offset < height) else (width, height)

                    mat_matrix = page[0][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]                      # materiality matrix region
                    
                    fn_out = page[2] + '_' + str(page[3]) + '.png'                                                      # output filename
                    
                    tmp = cv2.resize(mat_matrix, (int(mat_matrix.shape[1]/3), int(mat_matrix.shape[0]/3)))
                    cv2.imshow('Finding materiality matrix', tmp)

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # asking user if the cropped region is ok
                    while True:
                        choice = input('\nIs this crop ok? [Y/n] ')
                        # automatic crop is ok
                        if choice.lower()[0] == 'y':
                            cv2.imwrite(os.path.join(self.__out_path, fn_out), mat_matrix)
                            print('Materiality matrix image file was wrote in', os.path.join(self.__out_path, fn_out))
                            break
                        # user cropping
                        elif choice.lower()[0] == 'n':
                            resized_page = cv2.resize(page[0], resize)
                            roi = cv2.selectROI('Select the region of interest', resized_page)
                            mat_matrix = page[0][int(roi[1]*3.5):int(roi[1]*3.5) + int(roi[3]*3.5), int(roi[0]*3.5):int(roi[0]*3.5) + int(roi[2]*3.5)]
                            resize_mat_matrix = (int(mat_matrix.shape[1]/3.5), int(mat_matrix.shape[0]/3.5))
                            tmp = cv2.resize(mat_matrix, resize_mat_matrix)
                            cv2.imshow('Selected materiality matrix', tmp)
                            cv2.waitKey(0)

                            cv2.imwrite(os.path.join(self.__out_path, fn_out), mat_matrix)
                            print(fn_out, 'was wrote in', self.__out_path)
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

def main(pdf_path, language, debug):
    extr = PDFToImage(pdf_path, language, debug)
    extr.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sustineo_pdf2plot',
                                description='This program extracts materiality matrices in raster-format from pdf files.\
                                            \nAuthor: nebuchadneZZar01 (Michele Ferro)\
                                            \nGitHub: https://github.com/nebuchadneZZar01/',\
                                            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('pathname')

    parser.add_argument('-l', '--language', type=str, default='ita',\
                        help='language of the plot to extract (default="ita")')

    parser.add_argument('-d', '--debug-mode', action='store_true',\
                        help='activate the visualization of the various passes')

    args = parser.parse_args()

    if os.path.isfile(args.pathname):
        main(args.pathname, args.language, args.debug_mode)
    else:
        if os.path.isdir(args.pathname):
            for fn in os.listdir(args.pathname):
                complete_fn = os.path.join(args.pathname, fn)
                main(complete_fn, args.language, args.debug_mode)
        else:
            print('ERROR: File {fn} does not exist'.format(fn = args.pathname))