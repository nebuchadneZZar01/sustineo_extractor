import cv2
import numpy as np
import math
from plot_elements import Blob

# object class that crops the image, returning
# the plot-part and the legend-part in order to
# later manipulate them separately 
class Cropper:
    def __init__(self, image, debug_mode = False, scale_factor = float):
        self.__image = image
        self.__image_size = image.shape[0:2]
        self.__image_size = self.image_size[::-1]      

        self.__image_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        _, self.__work_image = cv2.threshold(self.__image_gray, 180, 255, cv2.THRESH_BINARY)

        self.__debug_mode = debug_mode
        
        if self.__debug_mode:
            self.__scale_size = (int(self.image_size[0]/scale_factor), int(self.image_size[1]/scale_factor))
            self.__image_debug = self.__image.copy()

            self.__scale_factor = scale_factor
            
            tmp_original = cv2.resize(self.__image, self.__scale_size)
            tmp_work = cv2.resize(self.__work_image, self.__scale_size)

            cv2.imshow('Finding image limits', tmp_original)
            cv2.imshow('Finding image limits', tmp_work)
            cv2.waitKey(1500)

    @property
    def image_size(self):
        return self.__image_size
    
    # finds all the blocks composing the plot
    # using the Hough transform
    def find_delimiters(self):
        edges = cv2.Canny(self.__work_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 450)

        self.__rows = []
        self.__columns = []

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
                    self.__rows.append(int(r_theta[0][0]))
                else:
                    self.__columns.append(int(r_theta[0][0]))

                if self.__debug_mode:
                    cv2.line(self.__image_debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        if self.__debug_mode:
            tmp = cv2.resize(self.__image_debug, self.__scale_size)

            cv2.imshow('Finding image limits', tmp)
            cv2.waitKey(1500)

    # finds the intersection point between lines
    # in order to detect the vertices that defines
    # the blocks composing the plot
    def find_intersections(self):
        tups = [(r, c) for r in self.__rows for c in self.__columns]
        
        if self.__debug_mode:
            for t in tups:
                cv2.circle(self.__image_debug, t, 3, (255, 0, 0), 3)

            tmp = cv2.resize(self.__image_debug, self.__scale_size)

            cv2.imshow('Finding image limits', tmp)
            cv2.waitKey(1500)

    # using a scale 1:3 (we know that the materiality matrices contains 3x3 blocks)
    # we find the vertices defining the entire plot
    def find_plot(self):
        self.find_delimiters()
        self.find_intersections()
        self.__rows.sort()
        self.__columns.sort(reverse=True)

        # LITTLE RECTANGLE IN THE PLOT
        # vertices delimiting the rectangle
        l_bottom_left = (self.__rows[0], self.__columns[0])                                 # first x coord, least y coord
        l_bottom_right = (self.__rows[-1], self.__columns[0])                               # least x coord, first y coord
        l_top_left = (self.__rows[0], self.__columns[-1])                                   # first x coord, least y coord

        # size of the rectangle's edges
        l_w = l_bottom_right[0] - l_bottom_left[0]
        l_h = l_bottom_left[1] - l_top_left[1]

        if self.__debug_mode:
            cv2.circle(self.__image_debug, l_bottom_left, 3, (0, 255, 0), 3)
            cv2.circle(self.__image_debug, l_bottom_right, 3, (0, 255, 0), 3)
            cv2.circle(self.__image_debug, l_top_left, 3, (0, 255, 0), 3)

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

        if self.__debug_mode:
            cv2.circle(self.__image_debug, b_bottom_left, 3, (255, 255, 0), 3)
            cv2.circle(self.__image_debug, b_top_left, 3, (255, 255, 0), 3)
            cv2.circle(self.__image_debug, b_bottom_right, 3, (255, 255, 0), 3)

            tmp = cv2.resize(self.__image_debug, self.__scale_size)
            cv2.imshow('Finding image limits', tmp)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
        
        return b_top_left, b_bottom_right

    def separate_image(self):
        top_left, bottom_right = self.find_plot()
        v_offset = 20                               # vertical offset
        h_offset = 200                              # horizontal offset
        w = top_left[0] - bottom_right[0] if (top_left[0] - bottom_right[0]) > 0 else -1 * (top_left[0] - bottom_right[0])
        h = bottom_right[1] - top_left[0] if (bottom_right[1] - top_left[0]) > 0 else -1 * (bottom_right[1] - top_left[0])
    
        plot = self.__image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # if the image has a legend on the left or on the right
        # the legend part will be composed by the intire original
        # image, but with the plot part filled with white
        if (self.image_size[0] - w) > h_offset:
            legend = self.__image.copy()
            cv2.rectangle(legend, top_left, bottom_right, (255, 255, 255), -1)
        else:
            legend = self.__image[bottom_right[1]:self.image_size[1], top_left[0]:bottom_right[0]]

        if self.__debug_mode:
            tmp_plot_r_size = (int(plot.shape[1]/self.__scale_factor), int(plot.shape[0]/self.__scale_factor))
            tmp_leg_r_size = (int(legend.shape[1]/self.__scale_factor), int(legend.shape[0]/self.__scale_factor))

            tmp_plot = cv2.resize(plot, tmp_plot_r_size)
            tmp_leg = cv2.resize(legend, tmp_leg_r_size)

            cv2.imshow('Plot OCR', tmp_plot)
            cv2.imshow('Legend OCR', tmp_leg)
            cv2.waitKey(1500)

        return plot, legend
    
class BlobCropper:
    def __init__(self, image, debug_mode = bool, scale_factor = float):
        self.__image = image
        self.__image_size = image.shape[0:2]
        self.__image_size = self.image_size[::-1]      

        self.__image_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        _, self.__work_image = cv2.threshold(self.__image_gray, 180, 255, cv2.THRESH_BINARY)

        self.__debug_mode = debug_mode
        
        if self.__debug_mode:
            self.__scale_size = (int(self.image_size[0]/scale_factor), int(self.image_size[1]/scale_factor))
            self.__image_debug = self.__image.copy()
            
            tmp_original = cv2.resize(self.__image, self.__scale_size)
            tmp_work = cv2.resize(self.__work_image, self.__scale_size)

            cv2.imshow('Finding image limits', tmp_original)
            cv2.imshow('Finding image limits', tmp_work)
            cv2.waitKey(1500)
    
    def __init_detector(self):
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 256

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 10000

        # Filter by Color (black=0)
        params.filterByColor = True
        params.blobColor = 0

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.maxCircularity = 1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.maxConvexity = 1

        # Filter by InertiaRatio
        params.filterByInertia = True
        params.minInertiaRatio = 0
        params.maxInertiaRatio = 1

        # Distance Between Blobs
        params.minDistBetweenBlobs = 0

        self.__blob_detector = cv2.SimpleBlobDetector_create(params)
        self.__blobboxes = []

    def __extract_shapes(self):
        _, shapes = cv2.threshold(self.image_gray, 240, 255, cv2.THRESH_BINARY)

        # we need to dilate the image in order to remove the lines
        # otherwise, the boxes crossed by the line won't be detected
        dilate_kernel = np.ones((3,3), np.uint8)
        dilated_shapes = cv2.dilate(shapes, dilate_kernel)
        
        keypoints = self.__blob_detector.detect(dilated_shapes)
        
        for keypoint in keypoints:
            cx = int(keypoint.pt[0])
            cy = int(keypoint.pt[1])
            s = keypoint.size
            r = int(math.floor(s/2))

            blob = Blob(cx, cy, r)
            self.__blobboxes.append(blob)

        if self.debug_mode:
            blob_detected = cv2.drawKeypoints(self.image_debug, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            tmp = cv2.resize(blob_detected, self.scale_size)

            cv2.imshow("Finding image limits", tmp)
            cv2.waitKey(0)

    def __find_legend(self):
        self.__init_detector()
        self.__extract_shapes()

        legend_blobs = []

        for bb_i in self.__blobboxes:
            for bb_j in self.__blobboxes:
                if ((bb_i.position[1] == bb_j.position[1] or \
                     bb_i.position[0] == bb_j.position[0]) and \
                     bb_i != bb_j):
                    legend_blobs.append(bb_i)

        i = 0
        while i < len(legend_blobs) - 1:
            if legend_blobs[i] == legend_blobs[i+1]:
                del legend_blobs[i+1] 

        print(len(legend_blobs))
        for lb in legend_blobs:
            print(lb.position)

    def separate_image(self):
        self.__find_legend()
        pass

    @property
    def image_size(self):
        return self.__image_size
    
    @property
    def image_debug(self):
        return self.__image_debug
    
    @property
    def image_gray(self):
        return self.__image_gray
    
    @property
    def debug_mode(self):
        return self.__debug_mode
    
    @property
    def scale_size(self):
        return self.__scale_size
