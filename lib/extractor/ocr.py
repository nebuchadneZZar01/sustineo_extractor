import cv2
import math
import numpy as np
import pandas as pd
from lib.extractor.plot_elements import TextBox, LabelBox, LabelBoxColorless, LegendBox, BlobBox
from pytesseract import pytesseract as pt

class OCR:
	"""Object defining the Optical Character Recognition; it externally calls a TesseractOCR routine using the pytesseract module.

	Keyword Arguments:
		- image -- Data of the input image file containing the plot
		- lang -- Language of the plot
		- scale_factor -- Determines the scale-down of the image visualization
		- debug_mode -- Option to visualize the shape detection in real-time 
	"""
	def __init__(self, image, lang: str, scale_factor: float, debug_mode: bool):
		self.__image = image
		self.__debug_mode = debug_mode
		self.__lang = lang						# language of the labels

		self.__image_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)

		self.__textboxes = []					# will contain the bounding boxes of every single word
		
		# image sizes will be divided by scale factor
		# to permit a better visualization on lower
		# resolution screens
		if debug_mode:
			self.__image_debug = self.__image.copy()
			self.__scale_factor = scale_factor if scale_factor != 0.0 else 1.0
			
			self.__scale_size = (int(self.__image.shape[1]/self.__scale_factor), int(self.__image.shape[0]/self.__scale_factor))

	@property
	def image_original(self):
		return self.__image

	@property
	def image_gray(self):
		return self.__image_gray

	@property
	def image_debug(self):
		return self.__image_debug

	@property
	def scale_size(self):
		return self.__scale_size

	@property
	def debug_mode(self):
		return self.__debug_mode

	@property
	def language(self):
		return self.__lang
	
	@property
	def textboxes(self):
		return self.__textboxes

	def show_image(self):
		pass

	def get_data(self):
		pass

class PlotOCR(OCR):
	"""Class defining the euristics later used following the different type of plots (OCR subclass).

	Keyword Arguments:
		- image -- Data of the input image file containing the plot
		- lang -- Language of the plot
		- scale_factor -- Determines the scale-down of the image visualization
		- debug_mode -- Option to visualize the shape detection in real-time 
	"""
	def __init__(self, image, lang = str, scale_factor = float, debug_mode = bool):
		super(PlotOCR, self).__init__(image, lang, scale_factor, debug_mode)
		self.__labelboxes = []					# will contain the bounding boxes of the entire labels

		_, self.work_image = cv2.threshold(self.image_gray, 185, 255, cv2.THRESH_BINARY)
	
	@property
	def labelboxes(self):
		return self.__labelboxes
	
	@labelboxes.setter
	def labelboxes(self, new_labelboxes):
		self.__labelboxes = new_labelboxes

	@property
	def work_image(self):
		return self.__work_image
	
	@work_image.setter
	def work_image(self, new_work_image):
		self.__work_image = new_work_image

	# function that calls the tesseract OCR
	def process_text(self):
		res = pt.image_to_data(self.work_image, lang=self.language, output_type = pt.Output.DICT)
		res = pd.DataFrame(res)
		res = res.loc[res['conf'] != -1]

		# verifies the color of the white and black pixels
		# in the binary image (considers the first understandable word): 
		# if the num of white pixels is higher than the
		# black ones, then the text is black (dark in the 
		# original image), otherwise is white
		lett_cnt = 0
		while lett_cnt < len(res):
			x = res.iloc[lett_cnt]['left']
			y = res.iloc[lett_cnt]['top']
			w = res.iloc[lett_cnt]['width']
			h = res.iloc[lett_cnt]['height']
			text = res.iloc[lett_cnt]['text']

			conf = int(res.iloc[lett_cnt]['conf'])

			if conf > 80:
				if len(text.strip(' ')) != 0:
					# the region of interest of the letter
					letter_roi = self.work_image[y:y+h, x:x+w]

					w_cnt = 0
					b_cnt = 0
					for r in letter_roi:
						for pixel in r:
							if pixel == 255: w_cnt += 1
							else: b_cnt += 1

					print('Num of white pixels:', w_cnt)
					print('Num of black pixels:', b_cnt)

					# converts to negative
					if w_cnt < b_cnt:
						print('In the threshold image, the text is white')
						print('Converting to negative')

						self.work_image = 255 - self.work_image
						dilatation_kernel = np.ones((2,2), np.uint8)
						self.work_image = cv2.dilate(self.work_image, dilatation_kernel)

						res = pt.image_to_data(self.work_image, lang='ita', output_type = pt.Output.DICT)
						res = pd.DataFrame(res)
						res = res.loc[res['conf'] != -1]

						if self.debug_mode:
							tmp = cv2.resize(self.work_image, self.scale_size)

							cv2.imshow('Plot OCR', tmp)
							cv2.waitKey(1500)
					break
			lett_cnt += 1

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

			if conf > 80:
				# display the confidence and text to our terminal
				# print("Confidence: {}".format(conf))
				# print("Text: {}".format(text))
				# print("")
				# strip out non-ASCII text so we can draw the text on the image
				# using OpenCV, then draw a bounding box around the text along
				# with the text itself
				text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

				if len(text) > 0:
					tb = TextBox((x, y), w, h, text)
					if self.debug_mode:
						cv2.rectangle(self.image_debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

					self.textboxes.append(tb)

class PlotOCR_Box(PlotOCR):
	"""Object useful to detect the plot-part of the image in box-type plots (PlotOCR subclass).

	Keyword Arguments:
		- image -- Data of the input image file containing the plot
		- lang -- Language of the plot
		- scale_factor -- Determines the scale-down of the image visualization
		- debug_mode -- Option to visualize the shape detection in real-time 
	"""
	def __init__(self, image, lang: str, scale_factor: float, debug_mode: bool):
		super(PlotOCR_Box, self).__init__(image, lang, scale_factor, debug_mode)
	
	# function that detects all the rectangles that contain the labels
	def __extract_labels(self):
		_, shapes = cv2.threshold(self.image_gray, 240, 255, cv2.THRESH_BINARY)

		# we need to dilate the image in order to remove the lines
		# otherwise, the boxes crossed by the line won't be detected
		dilate_kernel = np.ones((3,3), np.uint8)
		dilated_shapes = cv2.dilate(shapes, dilate_kernel)

		if self.debug_mode:
			tmp = cv2.resize(dilated_shapes, self.scale_size)

			cv2.imshow('Plot OCR', tmp)
			cv2.waitKey(1500)

		contours, _ = cv2.findContours(dilated_shapes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		i = 0
		for contour in contours:
			if i == 0:
				i = 1
				continue
	
			# function to approximate the shape
			approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
			
			if len(approx) >= 4 and len(approx) <= 8:
				# actual rectangles
				if len(approx) == 4:
					A = approx[0][0]		# upper left vertex
					D = approx[2][0]		# bottom right vertex

					lb = LabelBox(A, D)
					self.labelboxes.append(lb)
					
					if self.debug_mode:
						cv2.drawContours(self.image_debug, [contour], 0, (0, 255, 0), 2)
						cv2.circle(self.image_debug, A, 2, (255, 255, 0), 4)
						cv2.circle(self.image_debug, D, 2, (255, 255, 0), 4)

				# merged rectangles
				elif len(approx) == 8:
					r1_A = approx[3][0]			# upper-left
					r1_B = approx[4][0]			# upper-right
					r1_C = approx[2][0]			# bottom-left
					r1_D = approx[5][0]			# bottom-right

					r2_A = approx[1][0]
					r2_B = approx[6][0]
					r2_C = approx[0][0]
					r2_D = approx[7][0]

					# define rect 1
					lb1 = LabelBox(r1_A, r1_D)
					self.labelboxes.append(lb1)

					# define rect 2 
					lb2 = LabelBox(r2_A, r2_D)
					self.labelboxes.append(lb2)

					if self.debug_mode:
						cv2.drawContours(self.image_debug, [contour], 0, (0, 0, 255), 2)
						
						# draw rect 1
						cv2.rectangle(self.image_debug, r1_A, r1_D, (0, 255, 0), 3)

						# draw rect 2
						cv2.rectangle(self.image_debug, r2_A, r2_D, (0, 255, 0), 3)

						cv2.circle(self.image_debug, r1_A, 2, (255, 255, 0), 4)
						cv2.circle(self.image_debug, r1_B, 2, (255, 255, 0), 4)
						cv2.circle(self.image_debug, r1_C, 2, (255, 255, 0), 4)
						cv2.circle(self.image_debug, r1_D, 2, (255, 255, 0), 4)

						cv2.circle(self.image_debug, r2_A, 2, (0, 255, 255), 4)
						cv2.circle(self.image_debug, r2_B, 2, (0, 255, 255), 4)
						cv2.circle(self.image_debug, r2_C, 2, (0, 255, 255), 4)
						cv2.circle(self.image_debug, r2_D, 2, (0, 255, 255), 4)
				# for all figures that have more than 8 edges
				elif len(contour) > 8:
					len_contour = len(contour)
					# this loop deletes all vertices such that the euclidean distance
					# with the next one is lower than a certain offset
					offset = 10
					while len_contour > 8:
						i = 0
						while i < len_contour-1:
							point1 = contour[i][0]
							point2 = contour[i+1][0]
							x1 = point1[0]
							x2 = point2[0]
							y1 = point1[1]
							y2 = point2[1]
							dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
							if dist <= offset:
								contour = np.delete(contour, i, axis=0)
								len_contour -= 1
								break
							i += 1

					approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

					# then the shape can be divided into more two rectangles
					try:
						r1_A = approx[1][0]
						r1_D = approx[5][0]			# bottom-right
						
						lb1 = LabelBox(r1_A, r1_D)
						self.labelboxes.append(lb1)

						r2_B = approx[2][0]			# upper-right					
						r2_C = approx[4][0]			# upper-right					
						r2_D = approx[3][0]			# upper-left
						r2_A = (r2_C[0], r2_B[1])

						lb2 = LabelBox(r2_A, r2_D)
						self.labelboxes.append(lb2)

						if self.debug_mode:
							cv2.drawContours(self.__image_debug, [contour], 0, (0, 255, 255), 2)

							# draw rect 1
							cv2.rectangle(self.__image_debug, r1_A, r1_D, (0, 255, 0), 3)
							cv2.circle(self.__image_debug, r1_A, 2, (255, 255, 0), 4)
							cv2.circle(self.__image_debug, r1_D, 2, (255, 255, 0), 4)

							# draw rect 2
							cv2.rectangle(self.__image_debug, r2_A, r2_D, (0, 255, 0), 3)
							cv2.circle(self.__image_debug, r2_A, 2, (255, 255, 0), 4)
							cv2.circle(self.__image_debug, r2_B, 2, (255, 255, 0), 4)
							cv2.circle(self.__image_debug, r2_C, 2, (255, 255, 0), 4)
							cv2.circle(self.__image_debug, r2_D, 2, (255, 255, 0), 4)
					except:
						print('no vertices')

	# composes the labelboxes verifying if the
	# text is actually in that box
	def __compose_labelboxes(self):
		image_hsv = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2HSV)
		for lb in self.labelboxes:
			lb.color_rgb = (self.image_original[lb.position[1]+2, lb.position[0]+2])
			lb.color_hsv = (image_hsv[lb.position[1]+2, lb.position[0]+2])
			for tb in self.textboxes:
				lb.add_text_in_label(tb)

	# deletes all blank labelboxes
	def __verify_labelboxes(self):
		for lb in self.labelboxes.copy():
			if len(lb.label.strip(' ')) == 0 or len(lb.label) == 0:
				self.labelboxes.remove(lb)
			else:
				# removes all labelboxes containing
				# only single-character words
				# typically these are forms detected
				# as characters
				control = False
				for w in lb.label.split(' '):
					if len(w) == 1:
						control = True
					else:
						control = False
				if control:
					self.labelboxes.remove(lb)

		print(f'Labels extracted: {len(self.labelboxes)}\n')

		for lb in self.labelboxes:
			print(f'Position: ({lb.position[0]},{lb.position[1]})\n\
		 			Text: {lb.label}\nLabel length: {len(lb.label)}\n\
					Value: {lb.center}\n')
			if self.debug_mode:
				cv2.circle(self.image_debug, lb.center, 5, (0, 0, 255), 5)

	def process_image(self):
		self.__extract_labels()
		self.process_text()
		self.__compose_labelboxes()
		self.__verify_labelboxes()

	def get_data(self):
		return self.labelboxes

	# used in debug mode for the visualization
	def show_image(self):
		scaled_image = cv2.resize(self.image_debug, self.scale_size)
		scaled_threshold = cv2.resize(self.work_image, self.scale_size)
		scaled_grayscale = cv2.resize(self.image_debug, self.scale_size)
		
		cv2.imshow('Plot OCR', scaled_grayscale)
		cv2.imshow('Plot OCR', scaled_threshold)
		cv2.imshow('Plot OCR', scaled_image)
		cv2.waitKey(1500)
		cv2.destroyWindow('Plot OCR')

	# returns colors (in rgb format)
	# of the labelboxes in the plot
	def get_colors_rgb(self):
		colors = []

		for lb in self.labelboxes:
			if lb.color_rgb() not in colors:
				col = []
				for gr_lev in lb.color_rgb():
					gr = int(gr_lev)
					col.append(gr)
				col = tuple(col)
				colors.append(col)		

		return colors

	# returns colors (in opencv's hsv format)
	# of the labelboxes in the plot
	def get_colors_hsv(self):
		colors = []

		for lb in self.labelboxes:
			if lb.color_hsv not in colors:
				col = []
				for val in lb.color_hsv:
					val = int(val)
					col.append(val)
				col = tuple(col) 
				colors.append(col)
		
		return colors

class PlotOCR_Blob(PlotOCR):
	"""Object useful to detect the plot-part of the image in blob-type plots (PlotOCR subclass).

	Keyword Arguments:
		- image -- Data of the input image file containing the plot
		- lang -- Language of the plot
		- scale_factor -- Determines the scale-down of the image visualization
		- debug_mode -- Option to visualize the shape detection in real-time 
	"""
	def __init__(self, image, lang: str, scale_factor: float, debug_mode: bool):
		super(PlotOCR_Blob, self).__init__(image, lang, scale_factor, debug_mode)
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

	# using the blob detector, it detects
	# all the blobs in the plot
	def __extract_shapes(self):
		_, shapes = cv2.threshold(self.image_gray, 175, 255, cv2.THRESH_BINARY)

		# we need to dilate the image in order to remove the lines
		# otherwise, the boxes crossed by the line won't be detected
		dilate_kernel = np.ones((3,3), np.uint8)
		dilated_shapes = cv2.dilate(shapes, dilate_kernel)
		
		keypoints = self.__blob_detector.detect(dilated_shapes)
		
		for keypoint in keypoints:
			cx = int(keypoint.pt[0])			# center x coord
			cy = int(keypoint.pt[1])			# center y coord
			s = keypoint.size					# keypoint size
			r = int(math.floor(s/2))			# radius

			# ignoring all keypoints
			# that have neglectable
			# radius
			if r >= 5:
				blob = BlobBox(cx, cy, r)
				self.__blobboxes.append(blob)

		if self.debug_mode:
			blob_detected = cv2.drawKeypoints(self.image_debug, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			
			tmp_shapes = cv2.resize(shapes, self.scale_size)
			tmp_dilated = cv2.resize(dilated_shapes, self.scale_size)
			tmp_detected = cv2.resize(blob_detected, self.scale_size)
			
			cv2.destroyAllWindows()
			cv2.imshow("Plot OCR", tmp_shapes)
			cv2.imshow("Plot OCR", tmp_dilated)
			cv2.imshow("Plot OCR", tmp_detected)
			cv2.waitKey(1500)
	
	# deletes all blobs detected as
	# as strings not containing
	# alphabetic characters
	def __verify_textboxes(self):
		for tb in self.textboxes.copy():
			if not tb.text.isalpha():
				self.textboxes.remove(tb)

	# joins textboxes in order to have labels
	# including all the words from the topper-left
	# one to the bottom-right one
	def __compose_textboxes(self):
		label_list = []
		current_label = []
		current_row = []

		t_x = 25
		t_y = 30

		for i, tb in enumerate(self.textboxes):
			current_row.append(tb)

			if i < len(self.textboxes) - 1:
				next_tb = self.textboxes[i+1]

				dist_x = tb.distance_from_textbox_row(next_tb)

				if dist_x < t_x:
					continue

				dists_y = [current_row[0].distance_from_textbox_column(next_of_next) for next_of_next in self.textboxes[i+1:]]

				if any(dist_y < t_y for dist_y in dists_y):
					current_label.append(current_row)
					current_row = []
					continue

			current_label.append(current_row)
			label_list.append(current_label)
			current_label = []
			current_row = []
				
		for label in label_list:
			lb = LabelBoxColorless(label[0][0].top_left, label[-1][-1].bottom_right)
			self.labelboxes.append(lb)

			if self.debug_mode:
				cv2.rectangle(self.image_debug, label[0][0].top_left, label[-1][-1].bottom_right, (0,0,255), 4)

				tmp = cv2.resize(self.image_debug, self.scale_size)
				cv2.imshow('Plot OCR', tmp)
				cv2.waitKey(100)

		for i, lb in enumerate(self.labelboxes):
				lb.add_text_in_label(label_list[i])

	# deletes all empty labelboxes
	def __verify_labelboxes(self):
		for lb in self.labelboxes.copy():
			if len(lb.label.strip(' ')) <= 1:
				self.labelboxes.remove(lb)
			else:
				# removes all labelboxes containing
				# only single-character words
				control = False
				for w in lb.label.split(' '):
					if len(w) == 1:
						control = True
					else:
						control = False
				if control == True:
					self.labelboxes.remove(lb)
					

	# joins the blob with the nearest
	# (euclidean distance) labelbox
	def __compose_blobboxes(self):
		for bb in self.__blobboxes:
			dists = []
			# calculating all distances between
			# each blob and every labelboxes
			for lb in self.labelboxes:
				dist = bb.distance_from_point(lb.center)
				dists.append(dist)
			
			# taking the labelbox that has
			# minimum distance from the 
			# current blob
			dists = sorted(dists)
			for lb in self.labelboxes:
				if bb.distance_from_point(lb.center) == dists[0]:
					if not lb.taken:
						if len(bb.label) == 0:
							bb.label = lb.label
							lb.taken = True
							dists.remove(dists[0])
					else:
						continue

	# deletes all blobboxes having
	# empty text field 
	def __verify_blobboxes(self):
		for bb in self.__blobboxes.copy():
			if len(bb.label) <= 1:
				self.__blobboxes.remove(bb)

		print(f'Labels extracted: {len(self.__blobboxes)}\n')
		image_hsv = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2HSV)

		for bb in self.__blobboxes:
			bb.color_rgb = (self.image_original[bb.position[1], bb.position[0]])
			bb.color_hsv = (image_hsv[bb.position[1], bb.position[0]])
			print(f'Position: ({bb.position[0]},{bb.position[1]})\n\
		 			Text: {bb.label}\nLabel length: {len(bb.label)}\n')
			if self.debug_mode:
				cv2.circle(self.image_debug, bb.position, 5, (0, 0, 255), 5)

	def get_colors_hsv(self):
		colors = []

		for bb in self.__blobboxes:
			if bb.color_hsv not in colors:
				col = []
				for val in bb.color_hsv:
					val = int(val)
					col.append(val)
				col = tuple(col) 
				colors.append(col)
		
		return colors

	def process_image(self):
		self.__extract_shapes()
		self.process_text()
		self.__verify_textboxes()
		self.__compose_textboxes()
		self.__verify_labelboxes()
		self.__compose_blobboxes()
		self.__verify_blobboxes()

	def get_data(self):
		return self.__blobboxes

# OCR object class useful to detect
# the legend part of the image
class LegendOCR(OCR):
	"""Object useful to detect the plot-part of the image in blob-type plots (PlotOCR subclass).

	Keyword Arguments:
		- image -- Data of the image-part containing the legend
		- labelboxes -- List of all the labelboxes
		- lang -- Language of the plot
		- scale_factor -- Determines the scale-down of the image visualization
		- debug_mode -- Option to visualize the shape detection in real-time 
	"""
	def __init__(self, image, labelboxes: list, lang: str, scale_factor: float, debug_mode: bool):
		super(LegendOCR, self).__init__(image, lang, scale_factor, debug_mode)
		
		_, self.work_image = cv2.threshold(self.image_gray, 240, 255, cv2.THRESH_BINARY)
		self.colors_pos = []
		self.labelboxes = labelboxes

		if debug_mode:
			tmp = cv2.resize(self.work_image, self.scale_size)

			cv2.imshow('Legend OCR', tmp)
			cv2.waitKey(1500)
	
	# takes the text data
	# calling the OCR
	def __process_text(self):
		res = pt.image_to_data(self.work_image, lang=self.language, output_type = pt.Output.DICT)
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

			if conf > 80:
				text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

				if len(text) > 0:
					if text.isalpha():
						tb = TextBox((x, y), w, h, text)
						if self.debug_mode:
							cv2.rectangle(self.image_debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

						self.textboxes.append(tb)

	# using hsv color-space, it calls inRange()
	# function to detect color position
	def __get_colors_position(self, colors_hsv):
		i = 1
		image_hsv = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2HSV)
		for c in colors_hsv:
			mask = cv2.inRange(image_hsv, c, c)

			if self.debug_mode:
				tmp_hsv = cv2.resize(image_hsv, self.scale_size)
				tmp_mask = cv2.resize(mask, self.scale_size)

				cv2.imshow('Legend OCR', tmp_hsv)
				cv2.imshow('Legend OCR', tmp_mask)

				cv2.waitKey(1500)
				i += 1

			# finds the points where the mask is not zero
			points = cv2.findNonZero(mask)

			if points is not None:
				mean = np.mean(points, axis=0)
				x, y = (int(mean[0][0]), int(mean[0][1]))

				# taking rgb color
				c_rgb = tuple(self.image_original[y][x])[::-1]

				#print('{col} is in position ({_x}, {_y})\n'.format(col=c_rgb, _x=x, _y=y))
				self.colors_pos.append(((x, y), (c_rgb)))

		# verify if legend is disposed
		# in row or in column and
		# eventually sort it by y coord
		column_leg = False	
		for i, col in enumerate(self.colors_pos):
			if i < len(self.colors_pos) - 1:
				if col[0][0] == self.colors_pos[i+1][0][0]:
					column_leg = True
				else:
					continue

		if column_leg:
			self.colors_pos = sorted(self.colors_pos, key=lambda col: col[0][1])

	# joins the detected words
	# with the detected colors
	def __process_legend(self):
		if len(self.textboxes) > 0:
			label = self.textboxes[0].text
			for i in range(len(self.textboxes)):
				if i == len(self.textboxes)-1: break
				else:
					if self.textboxes[i].distance_from_textbox_row(self.textboxes[i+1]) < 15:
						label += ' ' + self.textboxes[i+1].text
					else:
						label += '\n' + self.textboxes[i+1].text

			legends_labels = label.split('\n')

			# take the first word of every level
			# and compute the distance between it
			# and the colored square: if it's lower
			# than a certain threshold, the square is
			# linked to the label via a LegendBox (to-do)
			self.__legendboxes = []
			for lb in legends_labels:
				for tb in self.textboxes:
					dists = []
					if lb.split(' ')[0] == tb.text:
						for c in self.colors_pos:
							c_pos = c[0]
							dist = tb.distance_from_point(c_pos)
							dists.append(dist)

						dists = sorted(dists)

						for c in self.colors_pos:
							c_pos = c[0]
							c_col = c[1]

							if tb.distance_from_point(c_pos) == dists[0]:
								if len(lb.split(' ')[0]) > 1:
									legend_box = LegendBox(c_pos)
									legend_box.color = c_col
									legend_box.label = lb
									dists.remove(dists[0])
									self.__legendboxes.append(legend_box)
			
			print("Extracted labels from legend:")
			for lb in self.__legendboxes:
				print(f"label: {lb.label}\ncolor: {lb.color}\n")
		else:
			self.__legendboxes = None

	def process_image(self, colors_hsv):
		self.__process_text()
		self.__get_colors_position(colors_hsv)
		self.__process_legend()

	def get_data(self):
		return self.__legendboxes

	def show_image(self):
		scaled_image = cv2.resize(self.image_debug, self.scale_size)
		scaled_threshold = cv2.resize(self.work_image, self.scale_size)
		scaled_grayscale = cv2.resize(self.image_gray, self.scale_size)
		
		cv2.imshow('Legend OCR', scaled_grayscale)
		cv2.imshow('Legend OCR', scaled_threshold)
		cv2.imshow('Legend OCR', scaled_image)
		cv2.waitKey(1500)
		cv2.destroyAllWindows()
