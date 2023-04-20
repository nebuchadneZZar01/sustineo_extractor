import cv2
import numpy as np
from scipy.spatial import distance
import sys

class Cropper:
    def __init__(self, template, image, debug_mode = False, scale_factor = float):
        self.image = image
        self.image_size = self.image.shape[0:2]
        self.image_size = self.image_size[::-1]             
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, self.work_image = cv2.threshold(self.image_gray, 180, 255, cv2.THRESH_BINARY)

        # used in the template matching
        self.rotation = [0, 90]

        self.debug_mode = debug_mode
        if self.debug_mode:
            self.scale_size = (int(self.image_size[0]/scale_factor), int(self.image_size[1]/scale_factor))
            self.image_debug = self.image.copy()

            self.scale_factor = scale_factor

            tmp = cv2.resize(self.work_image, self.scale_size)

            cv2.imshow('threshold', tmp)
            cv2.waitKey(0)
    
    # ---- TEMPLATE MATCHING PART (MAYBE WILL DELETE) ----
    # finds the arrow template
    def find_template(self):
        shape = self.template.shape
        w, h = shape[::-1]
        top_left = 0
        bottom_right = 0
        # scans every angle of rotation
        for i in range(len(self.rotation)):
            if self.rotation[i] == 0:
                template_rot = self.template
            elif self.rotation[i]:
                template_rot = cv2.rotate(self.template, cv2.ROTATE_90_CLOCKWISE)

            cv2.imshow('rotated', template_rot)

            res = cv2.matchTemplate(self.image_gray, template_rot, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            top_left_tmp = max_loc
            if self.rotation[i] == 0:
                bottom_right_tmp = (top_left_tmp[0] + w, top_left_tmp[1] + h)
                top_left = top_left_tmp
            elif self.rotation[i] == 90:
                bottom_right_tmp = (top_left_tmp[0] + h, top_left_tmp[1] + w)
                bottom_right = bottom_right_tmp

            # cv2.rectangle(self.image, top_left_tmp, bottom_right_tmp, (0,0,255), 2)

        return top_left, bottom_right
    # ---- END TEMPLATE MATCHING PART ----

    # finds all the blocks composing the plot
    # using the Hough transform
    def find_delimiters(self):
        edges = cv2.Canny(self.work_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 350)

        self.rows = []
        self.columns = []

        if lines is not None:
            for r_theta in lines:
                arr = np.array(r_theta[0], dtype=np.float64)
                r, theta = arr

                line_len = 3000

                a = np.cos(theta)
                b = np.sin(theta)

                x0 = a * r
                y0 = b * r

                x1 = int(x0 + line_len*(-b))
                y1 = int(y0 + line_len*(a))

                x2 = int(x0 - line_len*(-b))
                y2 = int(y0 - line_len*(a))
                                
                if r_theta[0][1] == 0:
                    self.rows.append(int(r_theta[0][0]))
                else:
                    self.columns.append(int(r_theta[0][0]))

                if self.debug_mode:
                    cv2.line(self.image_debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        if self.debug_mode:
            tmp = cv2.resize(self.image_debug, self.scale_size)

            cv2.imshow('lines detected', tmp)
            cv2.waitKey(0)

    # finds the intersection point between lines
    # in order to detect the vertices that defines
    # the blocks composing the plot
    def find_intersections(self):
        tups = [(r, c) for r in self.rows for c in self.columns]
        
        if self.debug_mode:
            for t in tups:
                cv2.circle(self.image_debug, t, 3, (255, 0, 0), 3)

            tmp = cv2.resize(self.image_debug, self.scale_size)

            cv2.imshow('intersections', tmp)
            cv2.waitKey(0)

    # using a scale 1:3 (we know that the materiality matrices contains 3x3 blocks)
    # we find the vertices defining the entire plot
    def find_plot(self):
        self.find_delimiters()
        self.find_intersections()
        self.rows.sort()
        self.columns.sort(reverse=True)

        # LITTLE RECTANGLE IN THE PLOT
        # vertices delimiting the rectangle
        l_bottom_left = (self.rows[0], self.columns[0])
        l_bottom_right = (self.rows[2], self.columns[0])
        l_top_left = (self.rows[0], self.columns[3])

        # size of the rectangle's edges
        l_w = l_bottom_right[0] - l_bottom_left[0]
        l_h = l_bottom_left[1] - l_top_left[1]

        if self.debug_mode:
            cv2.circle(self.image_debug, l_bottom_left, 3, (0, 255, 0), 3)
            cv2.circle(self.image_debug, l_bottom_right, 3, (0, 255, 0), 3)
            cv2.circle(self.image_debug, l_top_left, 3, (0, 255, 0), 3)

        # BIG RECTANGLE (THE PLOT)
        # size of the rectangle's edges
        b_h = 3 * l_h
        b_w = 3 * l_w

        # vertices delimiting the rectangle
        b_bottom_left = l_bottom_left
        # if the vertex calculated coordinates are beyond the image size
        # then it is set exactly on the border of the image
        b_top_left = (l_bottom_left[0], l_bottom_left[1] - b_h) if (l_bottom_left[1] - b_h) > 0 else (l_bottom_left[0], 0)
        b_bottom_right = (l_bottom_left[0] + b_w, l_bottom_left[1]) if (l_bottom_left[0] + b_w < self.image_size[0]) else (self.image_size[0], l_bottom_left[1])

        if self.debug_mode:
            cv2.circle(self.image_debug, b_bottom_left, 3, (255, 255, 0), 3)
            cv2.circle(self.image_debug, b_top_left, 3, (255, 255, 0), 3)
            cv2.circle(self.image_debug, b_bottom_right, 3, (255, 255, 0), 3)

            tmp = cv2.resize(self.image_debug, self.scale_size)
            cv2.imshow('plot', tmp)
            cv2.waitKey(0)
        
        return b_top_left, b_bottom_right

    def separate_image(self):
        top_left, bottom_right = self.find_plot()
        v_offset = 20                               # vertical offset
        h_offset = 200                              # horizontal offset
        w = top_left[0] - bottom_right[0] if (top_left[0] - bottom_right[0]) > 0 else -1 * (top_left[0] - bottom_right[0])
        h = bottom_right[1] - top_left[0] if (bottom_right[1] - top_left[0]) > 0 else -1 * (bottom_right[1] - top_left[0])
    
        plot = self.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        print(self.image_size[0] - w)

        # if the image has a legend on the left or on the right
        # the legend part will be composed by the intire original
        # image, but with the plot part filled with white
        if (self.image_size[0] - w) > h_offset:
            legend = self.image.copy()
            cv2.rectangle(legend, top_left, bottom_right, (255, 255, 255), -1)
        else:
            legend = self.image[bottom_right[1]:self.image.shape[1], top_left[0]:bottom_right[0]]

        if self.debug_mode:
            tmp_plot_r_size = (int(plot.shape[1]/self.scale_factor), int(plot.shape[0]/self.scale_factor))
            tmp_leg_r_size = (int(legend.shape[1]/self.scale_factor), int(legend.shape[0]/self.scale_factor))

            tmp_plot = cv2.resize(plot, tmp_plot_r_size)
            tmp_leg = cv2.resize(legend, tmp_leg_r_size)

            cv2.imshow('plot', tmp_plot)
            cv2.imshow('legend', tmp_leg)
            cv2.waitKey(0)

        return plot, legend

if __name__ == '__main__':
    image = cv2.imread('amadori2.png')
    template = cv2.imread('template.png')

    m = Cropper(template, image)
    m.find_delimiters()
    m.find_intersections()
    m.find_plot()