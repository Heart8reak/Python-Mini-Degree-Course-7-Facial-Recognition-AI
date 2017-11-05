import time
import os
import cv2
import numpy as np
from PIL import Image

def get_training_data(face_cascade, data_dir):
	images = []
	labels = []
	image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if not f.endswith('.wink')]

	for image_file in image_files:
		img = Image.open(image_file).convert('L')
		img = np.array(img, np.uint8)
		
		filename = os.path.split(image_file)[1]
		actual_person_number = int(filename.split('.')[0].replace('subject', ''))

		faces = face_cascade.detectMultiScale(img)
		for face in faces:
			x, y, w, h = face
			face_region = img[y:y+h, x:x+w]
			#face_region = cv2.resize(face_region, (150, 150))
			
			images.append(face_region)
			labels.append(actual_person_number)

	return images, labels

def evaluate(recognizer, face_cascade, data_dir):
	image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wink')]
	num_correct = 0

	for image_file in image_files:
		test_img = Image.open(image_file).convert('L')
		test_img = np.array(test_img, np.uint8)

		filename = os.path.split(image_file)[1]
		true_person_number = int(filename.split('.')[0].replace('subject', ''))

		faces = face_cascade.detectMultiScale(test_img, 1.05, 6)
		for face in faces:
			x, y, w, h = face
			face_region = test_img[y:y+h, x:x+w]
			#face_region = cv2.resize(face_region, (150, 150))

			predicted_person_number, confidence = recognizer.predict(face_region)

			if predicted_person_number == true_person_number:
				print "Correct classified %d with confidence %f" % (true_person_number, confidence)
				num_correct = num_correct + 1
			else:
				print "Incorrectly classified %d as %d" % (true_person_number, predicted_person_number)
				
	accuracy = num_correct / float(len(image_files)) * 100
	print "Accuracy: %.2f%%" % accuracy

def predict(recognizer, face_cascade, img):
	predictions = []
	faces = face_cascade.detectMultiScale(img, 1.05, 6)
	for face in faces:
		x, y, w, h = face
		face_region = img[y:y+h, x:x+w]
		#face_region = cv2.resize(face_region, (150, 150))
		start = time.time()
		predicted_person_number, confidence = recognizer.predict(face_region)
		print time.time() - start
		predictions.append((predicted_person_number, confidence))
	return predictions

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognition = cv2.createLBPHFaceRecognizer()

print "Getting training examples..."
images, labels = get_training_data(face_cascade, 'yalefaces')

print "Training..."
start = time.time()
face_recognition.train(images, np.array(labels))
print time.time() - start
print "Finished training!"
evaluate(face_recognition, face_cascade, 'yalefaces')

#img = cv2.imread('face.jpg', cv2.IMREAD_GRAYSCALE)
#print predict(face_recognition, face_cascade, img)

video_cap = cv2.VideoCapture(0)

while True:
	ret, frame = video_cap.read()
	#frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(gray, 1.05, 6)
	for face in faces:
		x, y, w, h = face
		face_region = gray[y:y+h, x:x+w]
		
		predicted_person_number, confidence = face_recognition.predict(face_region)
		
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
		cv2.putText(frame, str(predicted_person_number), (x,y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
			
	cv2.imshow('Running face recognition...', frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): break
	
video_cap.release()
cv2.destroyAllWindows()
		
		
