import cv2
import numpy as np
from pytesseract import pytesseract as pt

class WordsBox:
	def __init__(self, text = str, x = int, y = int, w = int, h = int):
		self.text = text
		self.x = x
		self.y = y
		self.w = w
		self.h = h

	def distance_next(next_word):
		p1 = (self.x + self.w, self.y+self.h)
		p2 = (next_word.x, self.y+self.h)

		return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[0])**2)

	def distance_row(next_row):
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

	def extract_labels(self):
		_, temp = cv2.threshold(self.image_gray, 240, 255, cv2.THRESH_BINARY)

		contours, _ = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		print(contours)

		i = 0
		for contour in contours:
			if i == 0:
				i = 1
			continue
	
			# cv2.approxPloyDP() function to approximate the shape
			approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
			
			# using drawContours() function
			cv2.drawContours(self.image_gray, [contour], 0, (0, 255, 0), 2)


	def process_text(self):
		self.extract_labels()

		res = pt.image_to_data(self.work_image, lang='ita', output_type = pt.Output.DICT)
		words = []

		for i in range(0, len(res["text"])):
			# extract the bounding box coordinates of the text region from
			# the current result
			x = res["left"][i]
			y = res["top"][i]
			w = res["width"][i]
			h = res["height"][i]
			# extract the OCR text itself along with the confidence of the
			# text localization
			text = res["text"][i]
			conf = int(res["conf"][i])

			if conf > 80:
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

			word = WordsBox(text, x, y, w, h)
			words.append(word)

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

# ocr = OCR('matrix.jpeg', 1.75)
# ocr.process_text()
# ocr.show_image()