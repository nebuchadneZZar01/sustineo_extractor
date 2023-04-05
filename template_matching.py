import cv2
import numpy as np

class Matcher:
    def __init__(self, template, image):
        self.image = image
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        self.rotation = [0, 90]

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

# image = cv2.imread('amadori2.png')
# template = cv2.imread('template.png')

# m = Matcher(template, image)
# m.separate_image()