from numpy import sqrt

# defines rectangular boxes (containing labels)
# in the plot
class Box:
    def __init__(self, top_left, bottom_right):
        self.top_left = tuple(top_left)
        self.bottom_right = tuple(bottom_right)
        self.width = bottom_right[0] - top_left[0]
        self.height = top_left[1] - bottom_right[1]

        self.center = (int((self.top_left[0] + self.bottom_right[0])/2), int((self.top_left[1] + self.bottom_right[1])/2)) 

    # computes the distance between
    # the center and a point
    def distance_from_point(self, point):
        dist = sqrt((point[0] - self.top_left[0])**2 + (point[1] - self.top_left[1])**2)

        return dist

    def get_position(self):
        return self.top_left

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height
    
    def get_center(self):
        return self.center  

# defines the bounding box of
# the words detected using OCR
class TextBox(Box):
    def __init__(self, position, width, height, text):
        super(TextBox, self).__init__(position, (position[0]+width, position[1]+height))
        self.position = tuple(position)
        self.width = width
        self.height = height
        self.text = text

        self.top_right = (self.bottom_right[0], self.top_left[1])

    def get_text(self):
        return self.text

    # defines if a textbox is contained in a label box shape
    # using is upper-left and bottom-right vertices  
    def in_box(self, box = Box):
        if (self.top_left[0] >= box.top_left[0] and self.top_left[1] >= box.top_left[1]) and \
            (self.bottom_right[0] <= box.bottom_right[0] and self.bottom_right[1] <= box.bottom_right[1]):
                return True
        else: 
            return False

    def distance_from_textbox(self, box = Box):
        dist = sqrt((box.top_left[0] - self.top_right[0])**2 + (box.top_left[1] - self.top_right[1])**2)

        return dist

# defines the labelboxes containing
# all the single words and forming
# the entire label
class LabelBox(Box):
    def __init__(self, top_left, bottom_right):
        super(LabelBox, self).__init__(top_left, bottom_right)
        self.label = ''
        self.color_rgb = None                                  # background color of the box
        self.color_hsv = None                                  # used to detect the color range
    
    # cuncatenate the word to the actual label if its 
    # bounding box is contained in the label box
    def add_text_in_label(self, tb=TextBox):
        if tb.in_box(self):
            self.label += tb.get_text() + ' '
        else:
            pass

    def set_color_hsv(self, color_hsv):
        self.color_hsv = tuple(color_hsv)

    def set_color_rgb(self, color_bgr):
        self.color_rgb = tuple(color_bgr[::-1])

    def get_color_rgb(self):
        return self.color_rgb

    def get_color_hsv(self):
        return self.color_hsv

    def get_label(self):
        return self.label