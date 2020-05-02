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

def recognize(image_path, database, model):
    
    encoding = img_to_encoding(image_path,model)
    
    min_dist = 200
    
    for (name, enc_list) in database.items():
        
        for db_enc in enc_list:
            dist = np.linalg.norm(db_enc-encoding)

            if dist<min_dist:
                min_dist = dist
                identity = name
    
    if min_dist > 0.58:
        print("Not in the database.")
    else:
        print ("it's " + str(identity))
        
    return min_dist, identity

def main():
    
    model = load_model('Model/face_recognition.h5',compile=False)
    cap=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    data_path='Temp/'
    
    if not os.path.isdir(data_path):
        os.mkdir(data_path)    

    print('\nDetecting face..')

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
                offset=25
                (x,y,w,h) = faces[-1]
                frame_crop=frame[y-offset:y+h+offset,x-offset:x+w+offset]
                frame_crop=cv2.resize(frame_crop,(96,96))
            except:
                continue
            
            print("\nFace captured. Enter 'q' to continue.")
            while True:
                cv2.namedWindow('Captured Face', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Captured Face', 300, 300)                
                cv2.imshow('Captured Face',frame_crop)
                key = cv2.waitKey(1) & 0xFF
                if key==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows
            break

    path = data_path+'detect'+'.png' 
    cv2.imwrite(path,frame_crop)
    
    enc_path='Data/Encodings/'
    
    if not os.path.exists(enc_path+'image_database.pkl'):
            database = {}
    else:    
        with open(enc_path+'image_database.pkl','rb') as f:
            database = pickle.load(f)
    
    min_dist,name = recognize(path,database,model)
    
    print(min_dist)

if __name__=='__main__':
    main()