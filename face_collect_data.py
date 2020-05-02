import cv2
import numpy as np
import os
import pickle
from keras.models import load_model
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def encodeImage(path,name='',single=False):
    data_path='Data/Encodings/'
    
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
        
    if single:
        embedding = img_to_encoding(path,model)
        if not os.path.exists(data_path+'image_database.pkl'):
            database = {}
        else:    
            with open(data_path+'image_database.pkl','rb') as f:
                database = pickle.load(f)
                
        if database.get(name)== None:
            database[name]=[]
        database[name].append(embedding)
        
        with open(data_path+'image_database.pkl','wb') as f:
            pickle.dump(database,f)

def takeImage():
    cap=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    data_path='Data/Images/'
    
    if not os.path.isdir(data_path):
        os.mkdir(data_path)    
    
    folder_name=input("Enter the name of the person: ")

    if not os.path.isdir(data_path+folder_name):
        os.mkdir(data_path+folder_name)

    image_path = data_path+folder_name+'/'

    print('Come in front of Webcam please')

    while True:
        ret,frame=cap.read()
        if ret==False:
            continue

        faces=face_cascade.detectMultiScale(frame,1.3,5)
        faces=sorted(faces,key=lambda f: f[2]*f[3])

        if len(faces)==0:
            continue
        else:
            try:  
                offset=30
                (x,y,w,h) = faces[-1]
                frame_crop=frame[y-offset:y+h+offset,x-offset:x+w+offset]
                frame_crop=cv2.resize(frame_crop,(96,96))
            except:
                continue
            print('Image Captured. Enter q to exit.')
            while True:
                cv2.namedWindow('Face', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Face', 300, 300)
                cv2.imshow('Face',frame_crop)
                key = cv2.waitKey(1) & 0xFF
                if key==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows
            break

    path = image_path+str(np.random.randint(1000000))+'.png'
    print(path)
    cv2.imwrite(path,frame_crop)
    encodeImage(path,folder_name,single=True)
    print(frame_crop.shape)
    print("Image saved Successfully")
    
if __name__=='__main__':
    
    model = load_model('Model/face_recognition.h5',compile=False)
    
    takeImage()