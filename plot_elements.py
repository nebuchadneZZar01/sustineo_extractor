# defines rectangular boxes (containing labels)
# in the plot
class Box:
    def __init__(self, top_left, bottom_right):
        self.top_left = tuple(top_left)
        self.bottom_right = tuple(bottom_right)
        self.width = bottom_right[0] - top_left[0]
        self.height = top_left[1] - bottom_right[1]

        self.center = (int((self.top_left[0] + self.bottom_right[0])/2), int((self.top_left[1] + self.bottom_right[1])/2)) 

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

# defines the labelboxes containing
# all the single words and forming
# the entire label
class LabelBox(Box):
    def __init__(self, top_left, bottom_right):
        super(LabelBox, self).__init__(top_left, bottom_right)
        self.label = ''
    
    # cuncatenate the word to the actual label
    # if its bounding box is contained in the label
    # box
    def add_text_in_label(self, tb=TextBox):
        if tb.in_box(self):
            self.label += tb.get_text() + ' '
        else:
            pass

    def get_label(self):
        return self.label

# text1 = TextBox((10, 10), 10, 10, 'ciao')
# text2 = TextBox((20, 20), 10, 10, 'come')
# text3 = TextBox((50, 30), 10, 10, 'va')
# text4 = TextBox((50, 40), 10, 10, 'la')
# text5 = TextBox((50, 50), 10, 10, 'vita')

# lb = LabelBox((5, 5), (100,100))
# lb.add_text_in_label(text1)
# lb.add_text_in_label(text2)
# lb.add_text_in_label(text3)
# lb.add_text_in_label(text4)
# lb.add_text_in_label(text5)

# print(lb.get_label())