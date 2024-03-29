"""OCR in Python using the Tesseract engine from Google
http://code.google.com/p/pytesser/
by Michael J.T. O'Kelly
V 0.0.1, 3/10/07"""

from PIL import Image
import subprocess
import cv2
import util
import errors
import numpy as np

tesseract_exe_name = r'C:\Program Files\Tesseract-OCR\tesseract'
# Name of executable to be called at command line
scratch_image_name = "temp.bmp" # This file must be .bmp or other Tesseract-compatible format
scratch_text_name_root = "temp" # Leave out the .txt extension
cleanup_scratch_flag = False  # Temporary files cleaned up after OCR operation

def call_tesseract(input_filename, output_filename):
	"""Calls external tesseract.exe on input file (restrictions on types),
	outputting output_filename+'txt'"""
	args = [tesseract_exe_name, input_filename, output_filename]
	proc = subprocess.Popen(args)
	retcode = proc.wait()
	if retcode!=0:
		errors.check_for_errors()

def image_to_string(im, cleanup = cleanup_scratch_flag):
	"""Converts im to file, applies tesseract, and fetches resulting text.
	If cleanup=True, delete scratch files after operation."""
	try:
		util.image_to_scratch(im, scratch_image_name)
		call_tesseract(scratch_image_name, scratch_text_name_root)
		text = util.retrieve_text(scratch_text_name_root)
	finally:
		if cleanup:
			util.perform_cleanup(scratch_image_name, scratch_text_name_root)
	return text

def image_file_to_string(filename, cleanup = cleanup_scratch_flag, graceful_errors=True):
	"""Applies tesseract to filename; or, if image is incompatible and graceful_errors=True,
	converts to compatible format and then applies tesseract.  Fetches resulting text.
	If cleanup=True, delete scratch files after operation."""
	try:
		try:
			call_tesseract(filename, scratch_text_name_root)
			text = util.retrieve_text(scratch_text_name_root)
		except errors.Tesser_General_Exception:
			if graceful_errors:
				im = Image.open(filename)
				text = image_to_string(im, cleanup)
			else:
				raise
	finally:
		if cleanup:
			util.perform_cleanup(scratch_image_name, scratch_text_name_root)
	return text
	

'''if __name__=='__main__':
	#im = cv2.imread('system.png')
	#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	#ret,thresh = cv2.threshold(imgray, 127, 255, 0)
	#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	#dilated = cv2.dilate(thresh,kernel,iterations = 5) # dilate
		
	
	im = Image.open('test.jpg')
	#im.thumbnail(size,Image.ANTIALIAS)
	#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	text = image_to_string(im)
	print text
	#try:
		#text = image_file_to_string('random.jpg', graceful_errors=False)
	#except errors.Tesser_General_Exception, value:
		#print "fnord.tif is incompatible filetype.  Try graceful_errors=True"
		#print value
	#text = image_file_to_string('random.jpg', graceful_errors=True)
	#print "fnord.tif contents:", text
	#text = image_file_to_string('test.png', graceful_errors=True)
	#print text
'''


