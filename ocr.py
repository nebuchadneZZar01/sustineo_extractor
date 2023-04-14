import cv2
import numpy as np
import pandas as pd
from plot_elements import *
from pytesseract import pytesseract as pt

class OCR:
	def __init__(self, image, scale_factor = float, debug_mode = False):
		self.image = image

		self.labelboxes = []				# will contain the bounding boxes of the entire labels
		self.textboxes = []					# will contain the bounding boxes of every single word

		self.datas = [ ]					# will contain the actually extracted data, which will then exported to a csv file

		self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		_, self.work_image = cv2.threshold(self.image_gray, 185, 255, cv2.THRESH_BINARY)
		erosion_kernel = np.ones((1,1), np.uint8)
		self.work_image = cv2.dilate(self.work_image, erosion_kernel)
		
		self.debug_mode = debug_mode

		if debug_mode:
			self.image_debug = self.image.copy()
			self.scale_factor = scale_factor if scale_factor != 0.0 else 1.0
			
			self.scale_size = (int(self.image.shape[1]/self.scale_factor), int(self.image.shape[0]/self.scale_factor))

	# function that detects all the rectangles that contain the labels
	def extract_labels(self):
		_, shapes = cv2.threshold(self.image_gray, 240, 255, cv2.THRESH_BINARY)

		# we need to dilate the image in order to remove the lines
		# otherwise, the boxes crossed by the line won't be detected
		dilate_kernel = np.ones((2,2), np.uint8)
		dilated_shapes = cv2.dilate(shapes, dilate_kernel)

		if self.debug_mode:
			tmp = cv2.resize(dilated_shapes, self.scale_size)

			cv2.imshow('shapes', tmp)

		contours, _ = cv2.findContours(dilated_shapes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		i = 0
		for contour in contours:
			if i == 0:
				i = 1
				continue
	
			# function to approximate the shape
			approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
			
			if len(approx) == 4 or len(approx) == 8:
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

	# function that call the tesseract OCR
	def process_text(self):
		self.extract_labels()

		res = pt.image_to_data(self.work_image, lang='ita', output_type = pt.Output.DICT)
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

	# composes the labelboxes verifying if the
	# text is actually in that box
	def compose_labelboxes(self):
		for lb in self.labelboxes:
			for tb in self.textboxes:
				lb.add_text_in_label(tb)

		self.verify_labelboxes()

	# deletes all blank labelboxes
	def verify_labelboxes(self):
		for lb in self.labelboxes.copy():
			if len(lb.label.strip(' ')) == 0 or len(lb.label) == 0:
				self.labelboxes.remove(lb)

		print('Labels extracted: {N}\n'.format(N = len(self.labelboxes)))

		for lb in self.labelboxes:
			lb.label = lb.label[:-1]
			print('Position: ({x},{y})\nText: {label}\nLabel length: {l}\nValue: {center}\n'.format(x = lb.get_position()[0],\
																								y = lb.get_position()[1],\
																								label = lb.get_label(),
																								l = len(lb.get_label()),
																								center = lb.get_center()))
			if self.debug_mode:
				cv2.circle(self.image_debug, lb.get_center(), 5, (0, 0, 255), 5)

	# function that makes a pandas dataframe
	# containing the extracted data
	def construct_dataset(self):
		for lb in self.labelboxes:
			self.datas.append((lb.label, lb.get_center()[0], lb.get_center()[1]))

		df = pd.DataFrame(self.datas, columns=('Label', 'GroupRel', 'StakeRel'))

		print('Showing the first rows of the dataset:')
		print(df.head())
		df.to_csv('data.csv')

		print('\nExtracted data was exported to data.csv')

	def extract_data(self):
		self.compose_labelboxes()
		self.construct_dataset()

	# used in debug mode for the visualization
	def show_image(self):
		scaled_image = cv2.resize(self.image_debug, self.scale_size)
		scaled_threshold = cv2.resize(self.work_image, self.scale_size)
		scaled_grayscale = cv2.resize(self.image_gray, self.scale_size)
		
		cv2.imshow('grayscale', scaled_grayscale)
		cv2.imshow('threshold ocr', scaled_threshold)
		cv2.imshow('ocr', scaled_image)
		cv2.waitKey(0)
	
	def get_image_work(self):
		return self.work_image