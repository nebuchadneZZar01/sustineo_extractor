import cv2
import numpy as np
import pandas as pd
from pytesseract import pytesseract as pt

class WordsBox:
	def __init__(self, text = str, x = int, y = int, w = int, h = int):
		self.text = text
		self.x = x
		self.y = y
		self.w = w
		self.h = h

	def distance_next(self, next_word):
		p1 = (self.x + self.w, self.y+self.h)
		p2 = (next_word.x, self.y+self.h)

		return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[0])**2)

	def distance_row(self, next_row):
		p1 = (self.x, self.y+self.h)
		p2 = (next_row.x, self.y)
		
		return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[0])**2)

class OCR:
	def __init__(self, image, scale_factor = float):
		self.image = image
		self.scale_factor = scale_factor	

		if self.image is None:
			print("ERROR: no image has been selected")
		else:
			self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
			_, self.work_image = cv2.threshold(self.image_gray, 185, 255, cv2.THRESH_BINARY)
			erosion_kernel = np.ones((1,1), np.uint8)
			self.work_image = cv2.erode(self.work_image, erosion_kernel)

	# function that detects all the rectangles that contain the labels
	def extract_labels(self):
		_, temp = cv2.threshold(self.image_gray, 240, 255, cv2.THRESH_BINARY)

		contours, _ = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		i = 0
		for contour in contours:
			if i == 0:
				i = 1
				continue
	
			# cv2.approxPloyDP() function to approximate the shape
			approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
			
			if len(approx) == 4 or len(approx) == 8:
				cv2.drawContours(self.image, [contour], 0, (0, 255, 0), 2)

				M = cv2.moments(contour)
				if M['m00'] != 0.0:
					x = int(M['m10']/M['m00'])
					y = int(M['m01']/M['m00'])

				cv2.circle(self.image, (x, y), 2, (255, 255, 0), 4)

	# functions that call the tesseract OCR
	def process_text(self):
		# self.extract_labels()

		res = pt.image_to_data(self.work_image, lang='ita', output_type = pt.Output.DICT)
		res = pd.DataFrame(res)
		res = res.loc[res['conf'] != -1]
		res.to_csv('image.csv')

		labels = []
		labels_objs = []

		label = ''
		label_objs = []
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
				word = WordsBox(text, x, y, w, h)
				# used to verify if word are in the same label box
				word_num = int(res.iloc[i]['word_num'])
				prev_word_num = int(res.iloc[i-1]['word_num']) if i >= 1 else None

				# display the confidence and text to our terminal
				# print("Confidence: {}".format(conf))
				# print("Text: {}".format(text))
				# print("")
				# strip out non-ASCII text so we can draw the text on the image
				# using OpenCV, then draw a bounding box around the text along
				# with the text itself
				text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
				cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)

				#cv2.circle(self.image, (x, y+h), 2, (0,0,255))
				if i != 0:
					if word_num == prev_word_num+1 or ((len(text) > 0) and text[0].islower()):
						label = label + ' ' + text
						label_objs.append(word)
					elif word_num == 1:
						if label != '':
							labels.append(label)
							labels_objs.append(label_objs)

						#label = ''
						label = text

						label_objs.clear()
						label_objs.append(word)
					else:
						label += text
						label_objs.append(word)
				
				# print(label)
				# print(label_objs)

		# print(labels)
		# print(labels_objs)

		for obj, label in zip(labels_objs, labels):
			print('objects: ', len(obj))
			
			for o in obj:
				print(o.text)
			
			break
			# words = label.split(' ')
			# print('words: ', len(words))
			# print('\n')


		# for i in range(len(words)):
		# 	if i + 1 < len(words):
		# 		if (words[i].distance_next(words[i+1]) < 5):
		# 			cv2.rectangle(self.image_gray, (words[i].x, words[i].y), (words[i+1].x + words[i+1].w, words[i+1].h + words[i+1].h), (0,255,0), 2)
		# 	else:
		# 		print('end')

	def show_image(self):
		if self.scale_factor != 0.0:
			image_size = self.image.shape
			new_size = (int(image_size[1]/self.scale_factor), int(image_size[0]/self.scale_factor))
			scaled_image = cv2.resize(self.image, new_size)
			scaled_threshold = cv2.resize(self.work_image, new_size)
			scaled_grayscale = cv2.resize(self.image_gray, new_size)
		
		cv2.imshow('grayscale', scaled_grayscale)
		cv2.imshow('threshold', scaled_threshold)
		cv2.imshow('ocr', scaled_image)
		cv2.waitKey(0)
	
	def get_image_work(self):
		return self.work_image