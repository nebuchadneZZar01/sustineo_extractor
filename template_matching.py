import cv2
import numpy as np
from scipy.spatial import distance
import sys

class Matcher:
    def __init__(self, template, image):
        self.image = image
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, self.work_image = cv2.threshold(self.image_gray, 180, 255, cv2.THRESH_BINARY)

        self.rotation = [0, 90]

        cv2.imshow('threshold', self.work_image)
        cv2.waitKey(0)

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

    def find_delimiters(self):
        edges = cv2.Canny(self.work_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 350)

        self.rows = []
        self.columns = []

        if lines is not None:
            print(len(lines))
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
                
                cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                if r_theta[0][1] == 0:
                    self.rows.append(int(r_theta[0][0]))
                else:
                    self.columns.append(int(r_theta[0][0]))
        
    
        # All the changes made in the input image are finally
        # written on a new image houghlines.jpg
        cv2.imshow('linesDetected.jpg', self.image)
        cv2.waitKey(0)

        # cv2.imshow('matched', self.image)
        # cv2.waitKey(0)
        # surf = cv2.SIFT_create()

        # kp1, des1 = surf.detectAndCompute(self.image_gray, None)
        # kp2, des2 = surf.detectAndCompute(self.template, None)

        # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
        # clusters = np.array([des2])
        # bf.add(clusters)

        # matches = bf.match(des2)
        # matches = sorted(matches, key = lambda x:x.distance)

        # res = cv2.drawMatches(self.image, kp1, self.template, kp2, matches[:10], None, flags=2)

        # cv2.imshow('matched', res)
        # cv2.waitKey(0)

    def find_intersections(self):
        tups = [(r, c) for r in self.rows for c in self.columns]
        for t in tups:
            print(t)
            cv2.circle(self.image, t, 3, (255, 0, 0), 3)
        
        cv2.imshow('intersections',image)
        cv2.waitKey(0)

    def find_plot(self):
        self.rows.sort()
        self.columns.sort(reverse=True)
        print(self.rows)
        print(self.columns)

        # little rectangle
        l_bottom_left = (self.rows[0], self.columns[0])
        l_bottom_right = (self.rows[2], self.columns[0])
        l_top_left = (self.rows[0], self.columns[3])

        l_w = l_bottom_right[0] - l_bottom_left[0]
        l_h = l_bottom_left[1] - l_top_left[1]

        cv2.circle(self.image, l_bottom_left, 3, (0, 255, 0), 3)
        cv2.circle(self.image, l_bottom_right, 3, (0, 255, 0), 3)
        cv2.circle(self.image, l_top_left, 3, (0, 255, 0), 3)

        print(l_w, l_h)

        # big rectangle
        b_h = 2 * l_h
        b_w = 2 * l_w

        print(b_w, b_h)

        b_bottom_left = l_bottom_left
        b_top_left = (l_bottom_left[0], l_bottom_left[1] - b_h)
        b_bottom_right = (l_bottom_left[0] + b_w, l_bottom_left[1])

        print(b_top_left)

        cv2.circle(self.image, b_bottom_left, 3, (255, 255, 0), 3)
        cv2.circle(self.image, b_top_left, 3, (255, 255, 0), 3)
        cv2.circle(self.image, b_bottom_right, 3, (255, 255, 0), 3)

        cv2.imshow('plot',image)
        cv2.waitKey(0)


    def separate_image(self):
        top_left, bottom_right = self.find_template()
        offset = 20
        w = top_left[0] - bottom_right[0]
        h = bottom_right[1] - top_left[0]
        plot = self.image[top_left[1]-offset:bottom_right[1], top_left[0]:bottom_right[0]+offset]
        legend = self.image[bottom_right[1]:self.image.shape[1], top_left[0]:bottom_right[0]]

        cv2.imshow('plot', plot)
        cv2.imshow('legend', legend)
        cv2.waitKey(0)

        return plot, legend

image = cv2.imread('amadori2.png')
template = cv2.imread('template.png')

m = Matcher(template, image)
m.find_delimiters()
m.find_intersections()
m.find_plot()