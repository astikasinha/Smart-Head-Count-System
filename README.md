# Smart-Head-Count-System
I have made a project for Mitzvah, where the requirement was to count the number of individuals entering a particular place,the camera will be fitted on the ceiling which will be connected to raspberry pi to process the ML model and count the number of individuals.
Here I have trained a YOLO model using 1000 images to detect faces and made a pipeline to load,track using bytetrack and display the count of people.
Bytetrack which is fast for real time use is used to track the people by assigning them unique ids.
OpenCv is used to access the webcam of the system.
Time is used to switch count to 0 after 60 seconds.
