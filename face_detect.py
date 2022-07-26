import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# 1st Photo
jeff_image = face_recognition.load_image_file("jeff.jpeg")
jeff_face_encoding = face_recognition.face_encodings(jeff_image)[0]

# 2nd Photo
elon_image = face_recognition.load_image_file("elon.jpeg")
elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

# 3rd Photo
oprah_image = face_recognition.load_image_file("oprah.jpeg")
oprah_face_encoding = face_recognition.face_encodings(oprah_image)[0]

# 4th Photo
jim_image = face_recognition.load_image_file("jim.jpeg")
jim_face_encoding = face_recognition.face_encodings(jim_image)[0]

# 5th Photo
malibu_image = face_recognition.load_image_file("malibu.jpeg")
malibu_face_encoding = face_recognition.face_encodings(malibu_image)[0]

Malala_image = face_recognition.load_image_file("Malala.jpeg")
Malala_face_encoding = face_recognition.face_encodings(Malala_image)[0]


baby_image = face_recognition.load_image_file("baby.jpeg")
baby_face_encoding = face_recognition.face_encodings(baby_image)[0]



# Array for the faces and names 
known_face_encodings = [
    jeff_face_encoding,
    elon_face_encoding,
    oprah_face_encoding,
    jim_face_encoding,
    Malala_face_encoding,
    malibu_face_encoding,
    baby_face_encoding,
]
known_face_names = [
    "Jeff Bezeos",
    "Elon Musk",
    "Oprah Winfrey",
    "Jim Carrey",
    "Malala Yousafzai",
    "Baby",
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    #This is being used to save time by using every other frame
    if process_this_frame:
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]
        
        #This is going to identify all the faces in the video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # This checks to see if the face in the camera matches any of the faces saved
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Outputs the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Creates a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Provides a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
