import os
import fitz
import cv2
import math
import numpy as np
import pandas as pd
import pytesseract as pt
import lib.pdf2plot.languages as languages
import gc

from lib.pdf2plot.document_page import DocumentPage
from tqdm import tqdm
from csv import DictWriter

DPI = 250                       # image DPI
ZOOM = DPI/72                   # image zoom

class PDFToImage:
    """Object for extracting images and plots from PDF format files.
    
    Keyword Arguments:
        - path -- Path to the PDF file to extract
        - language -- Language of the PDF file
        - headless -- Option to avoid GUI use
        - user_correction -- Option to manually correct the image extraction
        - paragraph -- Option to remove detected likely paragraphs in the extracted image
        - dataset_creation -- Option to create a dataset containing the extracted plot
        - debug_mode -- Option to visualize the shape detection in real-time
        - size_fatctor -- Determines the scale-down of the image visualization
    """
    def __init__(self, path: str, language: str, headless: bool, user_correction: bool, paragraph: bool, dataset_creation: bool, debug_mode: bool, size_factor: float):
        self.__path = path
        self.__filename = os.path.basename(self.__path)[:-4]

        self.__pdf_doc = fitz.open(path)
        self.__out_path = os.path.join(os.getcwd(), 'out', self.__filename)

        self.__out_img_path = os.path.join(self.__out_path, 'img')
        self.__out_plot_path = os.path.join(self.__out_path, 'img', 'plot')
        self.__out_plot_wo_par_path = os.path.join(self.__out_plot_path, 'no_par')
        self.__out_matrix_path = os.path.join(self.__out_path, 'img', 'm_matrix')

        self.__magnify = fitz.Matrix(ZOOM, ZOOM)
        
        self.__doc_pages = []

        self.__lang_dict = languages.LANGUAGE_DICT
        self.__lang = language

        self.__debug_mode = debug_mode
        self.__user_correction = user_correction
        self.__headless = headless
        self.__paragraph_removal = paragraph
        self.__dataset_creation = dataset_creation

        self.__size_factor = size_factor if size_factor != 0.0 else 1.0

        if self.__dataset_creation:
            self.__out_csv_annotations = os.path.join(os.getcwd(), 'out', 'annotations.csv')

        # extraction stats
        self.ex_materiality_mat_cnt = 0                 # total amount extracted materiality matrix
        self.ex_plot_cnt = 0                            # total amount extracted plots
        self.ex_plot_w_cnt = 0                          # amount possibly wrong extracted plots

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
    def out_plot_wo_par_path(self):
        return self.__out_plot_wo_par_path
    
    @property
    def out_matrix_path(self):
        return self.__out_matrix_path
    
    @property
    def out_csv_annotations(self):
        return self.__out_csv_annotations
    
    @property
    def doc_pages(self):
        return self.__doc_pages
    
    @doc_pages.deleter
    def doc_pages(self):
        del self.__doc_pages
    
    @property
    def lang_dict(self):
        return self.__lang_dict
    
    @property
    def lang(self):
        return self.__lang
    
    @property
    def headless(self):
        return self.__headless

    def __page_to_img(self):
        pbar = tqdm(range(self.__pdf_doc.page_count))
        pbar.set_description('Loading PDF')
        for page in pbar:
            text = self.__pdf_doc[page].get_text()
            
            # getting the text-image
            pix_text = self.__pdf_doc[page].get_pixmap(matrix=self.__magnify)
            pix_text.set_dpi(DPI, DPI)
            im_text = np.frombuffer(pix_text.samples, dtype=np.uint8).reshape(pix_text.h, pix_text.w, pix_text.n)
            im_text = np.ascontiguousarray(im_text[..., [2, 1, 0]])          # rgb to bgr

            # getting the textless-image
            # it is formed only of not-colored vector-type shapes
            paths = self.__pdf_doc[page].get_drawings()
            page_textless = self.__pdf_doc.new_page(width=self.__pdf_doc[page].rect.width, 
                                                    height=self.__pdf_doc[page].rect.height)
            
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
            im_textless = im_textless[..., [2, 1, 0]].copy()

            doc_page = DocumentPage(self.filename, page, im_text, im_textless, text)

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
        pbar = tqdm(self.doc_pages)                                                             # progress bar
        pbar.set_description('Extracting plots')
        for doc_page in pbar:
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

            resize = (int(doc_page.vector_page.shape[1]/self.__size_factor), 
                      int(doc_page.vector_page.shape[0]/self.__size_factor))

            if self.__debug_mode:
                tmp_res = cv2.resize(page_copy, resize)
                cv2.imshow('Finding interesting plots...', tmp_res)
                cv2.waitKey(1500)

            image_gray = cv2.cvtColor(page_copy, cv2.COLOR_BGR2GRAY)
            _, work_image = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)

            edges = cv2.Canny(work_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, math.pi/180, 700)
            circles = cv2.HoughCircles(work_image, cv2.HOUGH_GRADIENT, 1, 
                                       minDist=150, param1=100, param2=51, 
                                       minRadius=100, maxRadius=1000)

            rows = []
            columns = []

            if self.__debug_mode:
                tmp = doc_page.raster_page.copy()

            height, width, _ = doc_page.raster_page.shape
            fn_out = f'page_{doc_page.page_number}.png'                                                      # output filename

            # detecting if materiality matrix or not
            if self.lang_dict[self.lang] in doc_page.text.lower():
                self.ex_materiality_mat_cnt += 1
                final_path = self.out_matrix_path
            else:
                self.ex_plot_cnt += 1
                final_path = self.out_plot_path

            interesting_plot = None

            # detecting lines
            if lines is not None and circles is None:
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
                    cv2.imshow('Finding interesting plots...', tmp_res)
                    cv2.waitKey(1500)

                tups = [(r, c) for r in rows for c in columns]

                if self.__debug_mode:
                    for t in tups:
                        cv2.circle(tmp, t, 6, (255, 0, 0), 3)
                        tmp_res = cv2.resize(tmp, resize)

                    cv2.imshow('Finding interesting plots...', tmp_res)
                    cv2.waitKey(1500)

                rows.sort(); columns.sort(reverse=True)

                try:
                    # intersection delimiter points
                    if len(rows) == 0:
                        i_bottom_left = (0, columns[0])                                                               # limit x, first y coord
                        i_bottom_right = (work_image.shape[1], columns[0])                                            # origin x, first y coord
                        i_top_left = (0, columns[-1])                                                                 # limit x, least y coord
                        i_top_right = (work_image.shape[1], columns[-1])    
                    elif len(columns) == 0:
                        i_bottom_left = (rows[0], work_image.shape[0])                                                # first x coord, limit y
                        i_bottom_right = (rows[-1], work_image.shape[0])                                              # least x coord, origin y
                        i_top_left = (rows[0], 0)                                                                     # first x coord, origin y
                        i_top_right = (rows[-1], 0)
                    else:
                        i_bottom_left = (rows[0], columns[0])                                                               # first x coord, first y coord
                        i_bottom_right = (rows[-1], columns[0])                                                             # least x coord, first y coord
                        i_top_left = (rows[0], columns[-1])                                                                 # first x coord, least y coord
                        i_top_right = (rows[-1], columns[-1])                                                               # least x coord, first y coord

                    if doc_page.in_feasible_area(i_bottom_left) and doc_page.in_feasible_area(i_bottom_right) and \
                        doc_page.in_feasible_area(i_top_left) and doc_page.in_feasible_area(i_top_right):

                        if self.__debug_mode:
                            cv2.circle(tmp, i_bottom_left, 6, (0, 255, 0), 3)
                            cv2.circle(tmp, i_bottom_right, 6, (0, 255, 0), 3)
                            cv2.circle(tmp, i_top_left, 6, (0, 255, 0), 3)
                            cv2.circle(tmp, i_top_right, 6, (0, 255, 0), 3)

                            if not self.headless:
                                tmp_res = cv2.resize(tmp, resize)
                                cv2.imshow('Finding interesting plots...', tmp_res)
                                cv2.waitKey(1500)

                        # vertical offset
                        upper_offset = int(1/9 * doc_page.page_size[1])
                        lower_offset = int(1/10 * doc_page.page_size[1])   

                        # upper_offset = 350                              
                        # lower_offset = 300

                        # final cropped area delimiters
                        top_left = (0, i_top_left[1] - upper_offset) if (i_top_left[1] - upper_offset > 0) else (0, 0)
                        bottom_right = (width, i_bottom_right[1] + lower_offset) if (i_bottom_right[1] + lower_offset < height) else (width, height)

                        interesting_plot = doc_page.raster_page[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]                      # interesting plot region
                except:
                    print('Not-legit intersection found, skipping to next page...')
                    continue
            # detecting circles
            elif circles is not None:
                circles_hough = []
                for i in circles[0, :]:
                    center = (int(i[0]), int(i[1]))
                    # circle outline
                    radius = int(i[2])

                    circles_hough.append((center, radius))
                    if self.__debug_mode:
                        cv2.circle(tmp, center, 1, (0, 100, 100), 3)
                        cv2.circle(tmp, center, radius, (255, 0, 255), 3)
                
                min_x = circles_hough[0][0][0]
                min_y = circles_hough[0][0][1]
                max_x = 0
                max_y = 0

                for circle in circles_hough:
                    (x_center, y_center) = circle[0]

                    if x_center > max_x:
                        max_x = x_center

                    if y_center > max_y:
                        max_y = y_center

                    if x_center < min_x:
                        min_x = x_center

                    if y_center < min_y:
                        min_y = y_center

                for circle in circles_hough:
                    if min_y == circle[0][1]:
                        y_offset = int(circle[1] * 1.25)

                    if max_x == circle[0][0]:
                        x_offset = int(circle[1] * 1.25)

                top_x = min_x - x_offset
                top_y = min_y - y_offset

                bottom_x = max_x + x_offset
                bottom_y = max_y + y_offset

                if top_x < 0:
                    top_x = 0

                if top_y < 0:
                    top_y = 0

                if bottom_x > work_image.shape[1]:
                    bottom_x = work_image.shape[1]

                if bottom_y > work_image.shape[0]:
                    bottom_y = work_image.shape[0]

                top_left = (top_x, top_y)
                bottom_right = (bottom_x, bottom_y)

                if self.__debug_mode:
                    cv2.rectangle(tmp, top_left, bottom_right, (255, 0, 0), 3)

                    tmp_res = cv2.resize(tmp, resize)
                    cv2.imshow('Finding interesting plots...', tmp_res)
                    cv2.waitKey(1500)

                interesting_plot = doc_page.raster_page[top_left[1]:bottom_right[1], 
                                                        top_left[0]:bottom_right[0]]                      # interesting plot region
            else:
                continue
            
            # if user correction is enabled from cli
            if (self.__user_correction or self.__dataset_creation) and interesting_plot is not None:
                if not self.headless:
                    tmp = cv2.resize(interesting_plot, (int(interesting_plot.shape[1]/self.__size_factor), 
                                                        int(interesting_plot.shape[0]/self.__size_factor)))
                    
                    cv2.imshow('Found an interesting plot!', tmp)

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                # asking user if the cropped region is ok
                while True:
                    choice = input('\nIs this crop ok? [Y/n] ')
                    # no choice has been made
                    if len(choice) == 0:
                        print('Please make a choice!')
                        continue
                    else:
                        # automatic crop is ok
                        if choice.lower()[0] == 'y':
                            if not os.path.isdir(final_path):
                                os.makedirs(final_path)
                            cv2.imwrite(os.path.join(final_path, fn_out), interesting_plot)
                            break
                        # user cropping
                        elif choice.lower()[0] == 'n':
                            resized_page = cv2.resize(doc_page.raster_page, resize)
                            roi = cv2.selectROI('Select the region of interest', resized_page)

                            if roi[0] != 0 and roi[1] != 0 and roi[2] != 0 and roi[3] != 0:
                                interesting_plot = doc_page.raster_page[int(roi[1]*self.__size_factor):int(roi[1]*self.__size_factor) + int(roi[3]*self.__size_factor),
                                                                        int(roi[0]*self.__size_factor):int(roi[0]*self.__size_factor) + int(roi[2]*self.__size_factor)]
                                
                                resize_interesting_plot = (int(interesting_plot.shape[1]/self.__size_factor), 
                                                        int(interesting_plot.shape[0]/self.__size_factor))
                                
                                if not self.headless:
                                    tmp = cv2.resize(interesting_plot, resize_interesting_plot)
                                    cv2.imshow('Selected interesting plot', tmp)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()

                                if not os.path.isdir(final_path):
                                    os.makedirs(final_path)
                                cv2.imwrite(os.path.join(final_path, fn_out), interesting_plot)
                                break
                            else:
                                break
                        # invalid choice
                        else:
                            print('Invalid choice!')
                            continue
                
                # dataset creation will enable a new prompt
                if self.__dataset_creation:
                    while True:
                        class_label = input('Enter class label for this plot: ')

                        # no label has been inserted
                        if len(class_label) == 0:
                            print('Please insert a label!')
                            continue
                        else:
                            break

                    field_names = ['file_path', 'label']
                    row = {'file_path': os.path.join(os.path.relpath(final_path), fn_out),
                           'label': class_label}
                    
                    if not os.path.isfile(self.out_csv_annotations):
                        with open(self.out_csv_annotations, 'a') as csv_annotations:
                            dw = DictWriter(csv_annotations, fieldnames=field_names)
                            dw.writeheader()

                            dw.writerow(row)
                            csv_annotations.close()
                    else:
                        with open(self.out_csv_annotations, 'a') as csv_annotations:
                            dw = DictWriter(csv_annotations, fieldnames=field_names)

                            dw.writerow(row)
                            csv_annotations.close()            
            else:
                # if not enabled, image will be normally written  
                if interesting_plot is not None:
                    final_plot = interesting_plot
                    tmp = cv2.resize(interesting_plot, (int(interesting_plot.shape[1]/self.__size_factor), 
                                                        int(interesting_plot.shape[0]/self.__size_factor)))
                    
                    # searching text paragraphs
                    plot_gray = cv2.cvtColor(interesting_plot, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(plot_gray, 100, 255, cv2.THRESH_BINARY)[1]                   # first we threshold the image in order to thin the text

                    kernel = np.ones((5,5), np.uint8)
                    dilated_tresh = cv2.dilate(thresh, kernel)                                          # then we remove the text using a 5x5-kernel dilatation
                    dilated_tresh = cv2.dilate(dilated_tresh, kernel)                                   # (for two times)
                    opening = cv2.morphologyEx(dilated_tresh, cv2.MORPH_OPEN, kernel)                   # the opening is necessary to remove isolated pixels
                    negative = 255 - opening                                                            # we then find the negative image in order to
                    coords = cv2.findNonZero(negative)                                                  # find the non-zero points (where the plot is located)
                    x, y, w, h = cv2.boundingRect(coords)                                               # localizing the plot    

                    rect = interesting_plot[y:y+h, x:x+w]                                               # we then "fine-crop" the image
                                                                                                        # finally we calculate:
                    x_offset_crop = int(1/6 * interesting_plot.shape[1])                                # an horizontal offset (because of the lower width dimension)
                    y_offset_crop = int(1/10 * interesting_plot.shape[0])                               # and a vertical offset

                    if rect.shape != (0, 0, 3):                                                         # in order to crop a larger portion of the original image
                        final_rect = interesting_plot[(y - y_offset_crop):(y + h + y_offset_crop),      # which includes the final plot
                                                    (x - x_offset_crop):(x + w + x_offset_crop)]
                        
                        if final_rect.shape != (0, 0, 3):
                            # normal extraction mode
                            if not self.__paragraph_removal:
                                if not self.headless:
                                    cv2.imshow('Found an interesting plot!', tmp)
                                    cv2.waitKey(1500)

                                self.ex_plot_w_cnt += 1
                            # paragraph removal mode
                            else:
                                cv2.imshow('Interesting plot with paragraph', tmp)
                                cv2.waitKey(500)

                                if final_rect.size != 0:                                    
                                    try:
                                        final_plot = final_rect
                                        tmp = cv2.resize(final_rect, (int(final_rect.shape[1]/self.__size_factor), 
                                                                    int(final_rect.shape[0]/self.__size_factor)))
                                    
                                        cv2.imshow('Paragraphs removed!', tmp)
                                        cv2.waitKey(1500)

                                        if not os.path.isdir(self.out_plot_wo_par_path):
                                            os.makedirs(self.out_plot_wo_par_path)
                                        cv2.imwrite(os.path.join(self.out_plot_wo_par_path, fn_out), final_plot)
                                    except:
                                        print('There was an error: probably there wasn\'t actually any paragraph! Paragraph elimination aborted')
                                else:
                                    print('There was an error: probably there wasn\'t actually any paragraph! Paragraph elimination aborted')
                    else:
                        if not self.headless:
                            cv2.imshow('Found an interesting plot!', tmp)
                            cv2.waitKey(1500)
                    
                    if not self.headless:
                        cv2.destroyAllWindows()

                    # saving the final plot
                    if not os.path.isdir(final_path):
                        os.makedirs(final_path)
                    cv2.imwrite(os.path.join(final_path, fn_out), final_plot)

    def run(self):
        self.__page_to_img()
        self.__img_to_plot()

        # freeing memory in order
        # to prevent memory leaks
        del self.__pdf_doc
        del self.doc_pages
        gc.collect()

    def get_stats(self):
        total_amount = self.ex_materiality_mat_cnt + self.ex_plot_cnt
        ex_plot_c_cnt = self.ex_plot_cnt - self.ex_plot_w_cnt

        print(f'{total_amount} image extractions were made:')
        print(f'- {self.ex_materiality_mat_cnt} materiality matrices were extracted in {self.out_matrix_path}')
        print(f'- {self.ex_plot_cnt} plots were extracted in {self.out_plot_path} of which:')
        print(f'\t - {ex_plot_c_cnt} are extracted correctly')
        print(f'\t - {self.ex_plot_w_cnt} may require user intervention as were detected likely paragraphs (it could be useful to run the script in paragraph removal mode using the --paragraph [-p] argument)')