import cv2
import sys

#file path of the the cascade file using the run method of the file - second argument
cascPath = sys.argv[1]
#load the model with the path we just created into the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPath)

#create a video capture function
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
	
	#create the gray background with the web cam image displaying
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# find the faces in the full image displaying(gray back and the image of the web cam)
    faces = faceCascade.detectMultiScale(
		#set up some parameters for the cascade file
        gray, #file which to detect from
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
	
	#if the user clicks on q, stop the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
