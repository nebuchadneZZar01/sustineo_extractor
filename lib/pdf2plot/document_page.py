# class describing a document page and containing
# all the useful informations such as raster and 
# vector page, provenience filename and number 
class DocumentPage:
    def __init__(self, filename: str, page_number: int, raster_page, vector_page, page_text):
        self.__filename = filename
        self.__raster_page = raster_page
        self.__vector_page = vector_page
        self.__text = page_text
        self.__page_number = page_number

        self.__page_size = self.__raster_page.shape[0:2][::-1]
        # defining page margin by 1/12 * Width
        self.__margin_offset = int(self.__page_size[0]/12)

    @property
    def filename(self):
        return self.__filename
    
    @property
    def page_number(self):
        return self.__page_number

    @property
    def raster_page(self):
        return self.__raster_page

    @property
    def vector_page(self):
        return self.__vector_page

    @property
    def page_size(self):
        return self.__page_size
    
    @property
    def text(self):
        return self.__text
    
    @property
    def margin_offset(self):
        return self.__margin_offset

    # calculates the four borders areas
    # of the page, given the margin
    def __calc_feasible_area(self):
        t_left_delimiter = (self.margin_offset, self.margin_offset)
        t_right_delimiter = (self.page_size[0] - self.margin_offset, self.margin_offset)
        b_left_delimiter = (self.margin_offset, self.page_size[1] - self.margin_offset)
        b_right_delimiter = (self.page_size[0] - self.margin_offset, self.page_size[1] - self.margin_offset)

        return t_left_delimiter, t_right_delimiter, b_left_delimiter, b_right_delimiter

    # describes whether a point is in
    # the feasible area or not
    def in_feasible_area(self, point):
        t_left_delimiter, t_right_delimiter, b_left_delimiter, b_right_delimiter = self.__calc_feasible_area()

        if (point[0] > t_left_delimiter[0] and point[1] > t_left_delimiter[1]) and \
            (point[0] < t_right_delimiter[0] and point[1] > t_right_delimiter[1]) and \
            (point[0] > b_left_delimiter[0] and point[1] < b_left_delimiter[1]) and \
            (point[0] < b_right_delimiter[0] and b_right_delimiter[1]):
                return True
        else:
            return False