import streamlit as st
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

#Title of the application 
st.title("OpenCV Face Detection")

img_file_buffer = st.file_uploader("CHoose a file",type=['jpg','jpeg','png','gif'])

#Function for detecting faces
def detect_faces(net ,frame):
    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),[104,117,123],False,False)
    net.setInput(blob)
    detection = net.forward()
    return detection

def process_detection(frame,detection,conf_threshold=0.5):
    bboxes=[]
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detection[0,0,i,3]* frame_w)
            y1 = int(detection[0,0,i,4]* frame_h)
            x2 = int(detection[0,0,i,5]* frame_w)
            y2 = int(detection[0,0,i,6]* frame_h)
            bboxes.append([x1,y1,x2,y2])
            bb_line_thickness = max(1,int(round(frame_h/200)))
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),bb_line_thickness,cv2.LINE_8)
    return frame,bboxes        

#Functiom to load the DNN Model
@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile,modelFile)
    return net 



def get_img_download(img,filename,text):
    buffered = BytesIO()
    img.save(buffered,format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href= f'<a href="data:file/txt;base64,{img_str}" download = "{filename}">{text}</a>'
    return href

net = load_model()

if img_file_buffer is not None:
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()),dtype=np.uint8)
    img = cv2.imdecode(raw_bytes,cv2.IMREAD_COLOR)

    #create placeholder for input and output image
    placeholders = st.columns(2)

    placeholders[0].image(img,channels='BGR')
    placeholders[0].text("Input Image")
    
    #Creating slider for user to decide the threshold
    conf_threshold = st.slider("SET Confidence Threshold",min_value=0.0,max_value=1.0,step=0.01,value=0.5)

    detecttion = detect_faces(net,img)

    out_img,_ = process_detection(img,detecttion,conf_threshold)

    placeholders[1].image(out_img,channels='BGR')
    placeholders[1].text("Output Image")

    #Convert opencv image to PIL
    out_img = Image.fromarray(out_img[:,:,::-1])

    #create a link for downloading the output file
    st.markdown(get_img_download(out_img,'face_output.jpg','Download Output Image'),unsafe_allow_html=True)
