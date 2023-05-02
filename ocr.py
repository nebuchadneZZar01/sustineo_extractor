import os
import cv2
import math
import numpy as np
import pandas as pd
from plot_elements import *
from pytesseract import pytesseract as pt

class OCR:
	def __init__(self, image, lang = str, scale_factor = float, debug_mode = False):
		self.image = image
		self.debug_mode = debug_mode
		self.lang = lang						# language of the labels

		self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		
		# image sizes will be divided by scale factor
		# to permit a better visualization on lower
		# resolution screens
		if debug_mode:
			self.image_debug = self.image.copy()
			self.scale_factor = scale_factor if scale_factor != 0.0 else 1.0
			
			self.scale_size = (int(self.image.shape[1]/self.scale_factor), int(self.image.shape[0]/self.scale_factor))

	def process_text(self):
		pass

	def show_image(self):
		pass

class PlotOCR(OCR):
	def __init__(self, image, image_fn, lang = str, scale_factor = float, debug_mode = False):
		super(PlotOCR, self).__init__(image, lang, scale_factor, debug_mode)
		self.image_fn = image_fn

		self.labelboxes = []				# will contain the bounding boxes of the entire labels
		self.textboxes = []					# will contain the bounding boxes of every single word

		self.datas = [ ]					# will contain the actually extracted data, which will then exported to a csv file

		_, self.work_image = cv2.threshold(self.image_gray, 185, 255, cv2.THRESH_BINARY)
		
		
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
							cv2.drawContours(self.image_debug, [contour], 0, (0, 255, 255), 2)

							# draw rect 1
							cv2.rectangle(self.image_debug, r1_A, r1_D, (0, 255, 0), 3)
							cv2.circle(self.image_debug, r1_A, 2, (255, 255, 0), 4)
							cv2.circle(self.image_debug, r1_D, 2, (255, 255, 0), 4)

							# draw rect 2
							cv2.rectangle(self.image_debug, r2_A, r2_D, (0, 255, 0), 3)
							cv2.circle(self.image_debug, r2_A, 2, (255, 255, 0), 4)
							cv2.circle(self.image_debug, r2_B, 2, (255, 255, 0), 4)
							cv2.circle(self.image_debug, r2_C, 2, (255, 255, 0), 4)
							cv2.circle(self.image_debug, r2_D, 2, (255, 255, 0), 4)
					except:
						print('no vertices ')

	# function that calls the tesseract OCR
	def process_text(self):
		self.extract_labels()

		res = pt.image_to_data(self.work_image, lang='ita', output_type = pt.Output.DICT)
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
				if len(text.strip(' ')) != 0 :
					# region of interest of the letter
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

							cv2.imshow('negative', tmp)
							cv2.waitKey(0)

					break
				else:
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

	# composes the labelboxes verifying if the
	# text is actually in that box
	def compose_labelboxes(self):
		for lb in self.labelboxes:
			lb.set_color(self.image[lb.get_position()[1]+2, lb.get_position()[0]+2])
			for tb in self.textboxes:
				lb.add_text_in_label(tb)

		self.verify_labelboxes()

	def renormalize_x_group(self, value, norm_min, norm_max):
		x_values = [lb.get_center()[0] for lb in self.labelboxes]

		min_x = min(x_values)
		max_x = max(x_values)

		norm_value = (norm_max - norm_min) * (value - min_x)/(max_x - min_x) 

		return norm_value

	def renormalize_y_stake(self, value, norm_min, norm_max):
		y_values = [lb.get_center()[1] for lb in self.labelboxes]

		min_y = min(y_values)
		max_y = max(y_values)

		norm_value = (norm_max - norm_min) * (value - min_y)/(max_y - min_y) 

		norm_value -= norm_max
		if norm_value < 0:
			norm_value *= -1
	
		return norm_value

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
			group_value = self.renormalize_x_group(lb.get_center()[0], 0, 300)
			stake_value = self.renormalize_y_stake(lb.get_center()[1], 0, 300)
			self.datas.append((lb.label, group_value, stake_value, lb.color))

		df = pd.DataFrame(self.datas, columns=('Label', 'GroupRel', 'StakeRel', 'Color'))

		print('Showing the first rows of the dataset:')
		print(df.head())
		
		img_extension = self.image_fn[-3:len(self.image_fn)]

		out_dir = os.path.join('out')

		if not os.path.isdir(out_dir):
			os.mkdir(out_dir)

		csv_fn = self.image_fn.replace(img_extension, 'csv').replace('src/', '')
		out_path = os.path.join(out_dir, csv_fn)
		
		df.to_csv(out_path)
		print('\nExtracted data was exported to {fn}'.format(fn = out_path))

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

class LegendOCR(OCR):
	def __init__(self, image, lang, scale_factor, debug_mode):
		super(LegendOCR, self).__init__(image, lang, scale_factor, debug_mode)
		
		_, self.work_image = cv2.threshold(self.image_gray, 240, 255, cv2.THRESH_BINARY)
		erosion_kernel = np.ones((1,1), np.uint8)
		self.work_image = cv2.dilate(self.work_image, erosion_kernel)

		if debug_mode:
			tmp = cv2.resize(self.work_image, self.scale_size)

			cv2.imshow('legend shapes', tmp)
			cv2.waitKey(0)

	def process_forms(self):
		contours, _ = cv2.findContours(dilated_shapes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		i = 0
		for contour in contours:
			if i == 0:
				i = 1
				continue

			approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)