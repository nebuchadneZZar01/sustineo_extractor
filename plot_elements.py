from numpy import sqrt

# defines circular blobs
# in the plot
class Blob:
    def __init__(self, center_x, center_y, radius):
        self.__center_x = center_x
        self.__center_y = center_y
        self.__radius = radius

    # computes the distance between
    # the center and a point
    def distance_from_point(self, point):
        dist = sqrt((self.__center_x - point[0])**2 + (self.__center_y - point[1])**2)

        return dist

    @property
    def position(self):
        return (self.__center_x, self.__center_y)
    
    @property
    def center(self):
        return self.position
    
    @property
    def radius(self):
        return self.__radius


# defines rectangular boxes (containing labels)
# in the plot
class Box:
    def __init__(self, top_left, bottom_right):
        self.__top_left = tuple(top_left)
        self.__bottom_right = tuple(bottom_right)
        self.__bottom_left = (top_left[0], bottom_right[1])
        self.__width = bottom_right[0] - top_left[0]
        self.__height = top_left[1] - bottom_right[1]

        self.__center = (int((top_left[0] + bottom_right[0])/2), int((top_left[1] + bottom_right[1])/2)) 

    # computes the distance between
    # the center and a point
    def distance_from_point(self, point):
        dist = sqrt((self.__top_left[0] - point[0])**2 + (self.__top_left[1] - point[1])**2)

        return dist

    @property
    def position(self):
        return self.__top_left

    @property
    def top_left(self):
        return self.__top_left

    @property
    def bottom_right(self):
        return self.__bottom_right
    
    @property
    def bottom_left(self):
        return self.__bottom_left

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height
    
    @property
    def center(self):
        return self.__center  

# defines the bounding box of
# the words detected using OCR
class TextBox(Box):
    def __init__(self, position, width, height, text):
        super(TextBox, self).__init__(position, (position[0]+width, position[1]+height))
        self.__position = tuple(position)
        self.__width = width
        self.__height = height
        self.__text = text

        self.__top_right = (self.bottom_right[0], self.position[1])

    @property
    def text(self):
        return self.__text

    # defines if a textbox is contained in a label box shape
    # using is upper-left and bottom-right vertices  
    def in_box(self, box = Box):
        if (self.top_left[0] >= box.top_left[0] and self.top_left[1] >= box.top_left[1]) and \
            (self.bottom_right[0] <= box.bottom_right[0] and self.bottom_right[1] <= box.bottom_right[1]):
                return True
        else: 
            return False

    def distance_from_textbox_row(self, box = Box):
        dist = sqrt((box.top_left[0] - self.__top_right[0])**2 + (box.top_left[1] - self.__top_right[1])**2)

        return dist
    
    def distance_from_textbox_column(self, box = Box):
        dist = sqrt((box.top_left[0] - self.bottom_left[0])**2 + (box.top_left[1] - self.bottom_left[1])**2)
        
        return dist

# defines the labelboxes containing
# all the single words and forming
# the entire label
class LabelBox(Box):
    def __init__(self, top_left, bottom_right):
        super(LabelBox, self).__init__(top_left, bottom_right)
        self.__label = ''
        self.__color_rgb = '1'                                  # background color of the box
        self.__color_hsv = '1'                                  # used to detect the color range

        # defines how good the alignment is
        # compared to the diagonal
        self.__alignment = 0
    
    @property
    def color_rgb(self):
        return self.__color_rgb
    
    @color_rgb.setter
    def color_rgb(self, color_bgr):
        self.__color_rgb = tuple(color_bgr[::-1])   

    @property
    def color_hsv(self):
        return self.__color_hsv

    @color_hsv.setter
    def color_hsv(self, color_hsv):
        self.__color_hsv = tuple(color_hsv)

    @property
    def label(self):
        return self.__label

    @property
    def alignment(self):
        return self.__alignment

    @alignment.setter
    def alignment(self):
        pass

    # cuncatenate the word to the actual label if its 
    # bounding box is contained in the label box
    def add_text_in_label(self, tb=TextBox):
        if tb.in_box(self):
            self.__label += tb.text + ' '
        else:
            pass

# defines labelboxes without color
# used to identify labels that have
# to be joined with the detected blobs
class LabelBoxColorless(Box):
    def __init__(self, top_left, bottom_right):
        super(LabelBoxColorless, self).__init__(top_left, bottom_right)
        self.__label = ''
        self.__center = (int((top_left[0] + bottom_right[0])/2), int((top_left[1] + bottom_right[1])/2)) 
        # defines if labelbox has been already choosen for selection
        self.__taken = False

    @property
    def label(self):
        return self.__label
    
    @property
    def center(self):
        return self.__center
    
    @property
    def taken(self):
        return self.__taken
    
    @taken.setter
    def taken(self, boolean):
        self.__taken = boolean
    
    def distance_from_point(self, point):
        dist = sqrt((point[0] - self.center[0])**2 + (point[1] - self.center[1])**2)

        return dist
    
    def add_text_in_label(self, list_label):
        label = ''
        for row in list_label:
            for word in row:
                label += ' ' + word.text
        
        label = label[1:].lower()
        label = label.capitalize()

        self.__label = label

# blobs including the text and
# and the color data
# center equals the position
class BlobBox(Blob):
    def __init__(self, center_x, center_y, radius):
        super().__init__(center_x, center_y, radius)
        self.__label = ''
        self.__color_rgb = '1'                                  # background color of the box
        self.__color_hsv = '1'                                  # used to detect the color range

        # defines how good the alignment is
        # compared to the diagonal
        self.__alignment = 0

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, list_label):
        self.__label = ''.join(list_label)
        
    @property
    def color_rgb(self):
        return self.__color_rgb
    
    @color_rgb.setter
    def color_rgb(self, color_bgr):
        self.__color_rgb = tuple(color_bgr[::-1])   

    @property
    def color_hsv(self):
        return self.__color_hsv

    @color_hsv.setter
    def color_hsv(self, color_hsv):
        self.__color_hsv = tuple(color_hsv)

    @property
    def alignment(self):
        return self.__alignment

    @alignment.setter
    def alignment(self):
        pass

##########
# LEGEND #
##########

# defines the block containing a legend label
# with its color, its label and starting from
# the upper-left vertex
class LegendBox(Box):
    def __init__(self, top_left):
        super(LegendBox, self).__init__(top_left, bottom_right = (0,0))
        self.__label = ''
        self.__color = None
    
    @property
    def label(self):
        return self.__label

    @property
    def color(self):
        return self.__color

    @label.setter
    def label(self, new_label):
        self.__label = new_label

    @color.setter
    def color(self, new_color):
        self.__color = new_color