Python 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license()" for more information.
import cv2 as cv
haar_file = cv.CascadeClassifier('home/user/Desktop/haarcascade_frontalface_default.xml')
img = cv.imread('/home/user/Documents/documentations/pass photo.jpg')
img
array([[[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       ...,

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]]], dtype=uint8)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray
array([[135, 135, 135, ..., 135, 135, 135],
       [135, 135, 135, ..., 135, 135, 135],
       [135, 135, 135, ..., 135, 135, 135],
       ...,
       [ 31,  31,  31, ..., 135, 135, 135],
       [ 31,  31,  31, ..., 135, 135, 135],
       [ 31,  31,  31, ..., 135, 135, 135]], dtype=uint8)
faces = haar_file.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#6>", line 1, in <module>
cv2.error: OpenCV(4.10.0) /io/opencv/modules/objdetect/src/cascadedetect.cpp:1689: error: (-215:Assertion failed) !empty() in function 'detectMultiScale'

gray.shape
(1155, 959)
gray.min
<built-in method min of numpy.ndarray object at 0x79050c1bf210>
gray.min()
2
gray.max()
244
len(gray)
1155
gray.size()
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#12>", line 1, in <module>
TypeError: 'int' object is not callable
gray.size
1107645
faces = haar_file.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#14>", line 1, in <module>
cv2.error: OpenCV(4.10.0) /io/opencv/modules/objdetect/src/cascadedetect.cpp:1689: error: (-215:Assertion failed) !empty() in function 'detectMultiScale'

print(haar_file.empty())
True
haar_file = cv.CascadeClassifier('home/user/Desktop/haarcascade_frontalface_default.xml')
if haar_file.empy():
    print("file was not loaded correctly")

    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#19>", line 1, in <module>
AttributeError: 'cv2.CascadeClassifier' object has no attribute 'empy'. Did you mean: 'empty'?
if haar_file.empty():
    print("file was not loaded correctly")

    
file was not loaded correctly
haar_file = cv.CascadeClassifier("/home/user/Desktop/haarcascade_frontalface_default.xml")
haar_file.empty()
False
faces = haar_file.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
for (x,y,w,h) in faces :
    cv.rectangel(gray,(x,y),(x+w,y+h),-1,(0,255,0),1.5)

    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#27>", line 2, in <module>
AttributeError: module 'cv2' has no attribute 'rectangel'
for (x,y,w,h) in faces :
    cv.rectangle(gray,(x,y),(x+w,y+h),-1,(0,255,0),1.5)

    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#29>", line 2, in <module>
cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'rectangle'
> Overload resolution failed:
>  - Argument 'thickness' is required to be an integer
>  - Argument 'thickness' is required to be an integer
>  - Can't parse 'rec'. Expected sequence length 4, got 2
>  - Can't parse 'rec'. Expected sequence length 4, got 2

for (x,y,w,h) in faces :
    cv.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),1.5)

    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#31>", line 2, in <module>
cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'rectangle'
> Overload resolution failed:
>  - Argument 'thickness' is required to be an integer
>  - Argument 'thickness' is required to be an integer
>  - Can't parse 'rec'. Expected sequence length 4, got 2
>  - Can't parse 'rec'. Expected sequence length 4, got 2

print(faces)
[[288 270 392 392]]
for (x,y,w,h) in faces:
    print(x,y,w,h)

    
288 270 392 392
for (x,y,w,h) in faces :
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1.5)

    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#37>", line 2, in <module>
cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'rectangle'
> Overload resolution failed:
>  - Argument 'thickness' is required to be an integer
>  - Argument 'thickness' is required to be an integer
>  - Can't parse 'rec'. Expected sequence length 4, got 2
>  - Can't parse 'rec'. Expected sequence length 4, got 2

for (x,y,w,h) in faces :
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    
array([[[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       ...,

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]]], dtype=uint8)
cv.imwrite(img,'/home/user/Desktop/image.jpg')
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#40>", line 1, in <module>
cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'imwrite'
> Overload resolution failed:
>  - Expected 'filename' to be a str or path-like object
>  - Expected 'filename' to be a str or path-like object

cv.imwrite('image.jpg',img)
True
cv.imwrite('/home/user/Desktop/image.jpg',img)
True
for (x,y,w,h) in faces :
    for (x,y,w,h) in faces :
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        
array([[[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       ...,

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]]], dtype=uint8)
for (x,y,w,h) in faces :
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)

    
array([[[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       ...,

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]]], dtype=uint8)
cv.imwrite('/home/user/Desktop/image.jpg',img)
True
for (x,y,w,h) in faces :
    img_roi = img[y:y+h,x:x+w]
    labels,confidence = haar_file.predict(img_roi)

    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#52>", line 3, in <module>
AttributeError: 'cv2.CascadeClassifier' object has no attribute 'predict'
for (x,y,w,h) in faces :
    img_roi = img[y:y+h,x:x+w]

    
img_roi
array([[[  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0],
        ...,
        [  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0]],

       [[  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0],
        ...,
        [  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0]],

       [[  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0],
        ...,
        [  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0]],

       ...,

       [[  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0],
        ...,
        [213, 188,   0],
        [  0, 255,   0],
        [  0, 255,   0]],

       [[  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0],
        ...,
        [  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0]],

       [[  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0],
        ...,
        [  0, 255,   0],
        [  0, 255,   0],
        [  0, 255,   0]]], dtype=uint8)
cv.imwrite('/home/user/Desktop/roi.jpg',img_roi)
True
img1 = cv.imread('/home/user/Pictures/2persons.jpg',0)
img
array([[[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[213, 188,   0],
        [213, 188,   0],
        [213, 188,   0],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       ...,

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]],

       [[ 79,  30,  16],
        [ 79,  30,  16],
        [ 79,  30,  16],
        ...,
        [213, 188,   0],
        [213, 188,   0],
        [213, 188,   0]]], dtype=uint8)
faces = haar_file.detectMultiScale(img1,scaleFactor = 1.1, minNeighbors = 5)
print(faces)
[[ 53  47  76  76]
 [248  53  73  73]]
len(faces)
2
for i in range(len(faces)):
    for (x,y,w,h) in faces[i] ;
    
SyntaxError: incomplete input
for i in range(len(faces)):
    for (x,y,w,h) in faces[i] :
        cv.rectangle(img1,(x,y),(x+w,y+h), (255,0,0),4)

        
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#66>", line 2, in <module>
TypeError: cannot unpack non-iterable numpy.int32 object
faces[0]
array([53, 47, 76, 76], dtype=int32)
faces[1]
array([248,  53,  73,  73], dtype=int32)
for i in range(len(faces)):
    for (x,y,w,h) in faces[i] :
        cv.rectangle(img1,(x,y),(x+w,y+h), (255,0,0),4)

        
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#70>", line 2, in <module>
TypeError: cannot unpack non-iterable numpy.int32 object
for (x,y,w,h) in faces[0]:
    print(x,y,w,h)

    
Traceback (most recent call last):
  File "/usr/lib/python3.10/idlelib/run.py", line 578, in runcode
    exec(code, self.locals)
  File "<pyshell#73>", line 1, in <module>
TypeError: cannot unpack non-iterable numpy.int32 object
for (x,y,w,h) in faces:
    print(x,y,w,h)

    
53 47 76 76
248 53 73 73
faces.shape[1]
4
faces.shape
(2, 4)
faces.shape[0]
2
for i in range(faces.shape[0]:
               
SyntaxError: incomplete input
print(faces[0].reshape((1,4)))
               
[[53 47 76 76]]
>>> for i in range(faces.shape[0]):
...     for (x,y,w,h) in faces[i].reshape((1,4)):
...         cv.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),4)
... 
...                
array([[238, 238, 238, ..., 238, 238, 238],
       [238, 238, 238, ..., 238, 238, 238],
       [238, 238, 238, ..., 238, 238, 238],
       ...,
       [238, 238, 238, ..., 238, 238, 238],
       [238, 238, 238, ..., 238, 238, 238],
       [238, 238, 238, ..., 238, 238, 238]], dtype=uint8)
array([[238, 238, 238, ..., 238, 238, 238],
       [238, 238, 238, ..., 238, 238, 238],
       [238, 238, 238, ..., 238, 238, 238],
       ...,
       [238, 238, 238, ..., 238, 238, 238],
       [238, 238, 238, ..., 238, 238, 238],
       [238, 238, 238, ..., 238, 238, 238]], dtype=uint8)
>>> cv.imwrite('/home/user/Desktop/img1.jpg',img1)
...                
True
>>> for i in range(faces.shape[0]):
...     for (x,y,w,h) in faces[i].reshape((1,4)):
...         face_roi = img1[y:y+h,x:x+w]
...         cv.imwrite('%s.jpg'%(i),face_roi)
... 
...                
True
True
>>> for i in range(faces.shape[0]):
...     for (x,y,w,h) in faces[i].reshape((1,4)):
...         face_roi = img1[y:y+h,x:x+w]
...         cv.imwrite('/home/user/Desktop/%s.jpg'%(i),face_roi)
... 
...                
True
True
