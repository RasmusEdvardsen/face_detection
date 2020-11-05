import cv2
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import non_max_suppression_fast
from keras.models import load_model

def rcnn(image,base_model_name):
    model = load_model(base_model_name)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    results = ss.process()
    copy = image.copy()
    copy2 = image.copy()
    positive_boxes = []
    probs = []

    print(len(results))
    for box in results:
        x1 = box[0]
        y1 = box[1]
        x2 = box[0]+box[2]
        y2 = box[1]+box[3]

        roi = image.copy()[y1:y2,x1:x2]
        roi = cv2.resize(roi,(200,300))
        roi_use = roi.reshape((1,200,300,3))
        class_pred = model.predict_classes(roi_use)[0][0]
        if class_pred == 1:
            prob = model.predict(roi_use)[0][0]
            print(prob)
            if prob > 0.7:
                positive_boxes.append([x1,y1,x2,y2])
                probs.append(prob)
                cv2.rectangle(copy2,(x1,y1),(x2,y2),(255,0,0),5)

    cleaned_boxes = non_max_suppression_fast(np.array(positive_boxes),0.1,probs)
    total_boxes = 0
    print(len(cleaned_boxes), "hello")
    for clean_box in cleaned_boxes:
        clean_x1 = clean_box[0]
        clean_y1 = clean_box[1]
        clean_x2 = clean_box[2]
        clean_y2 = clean_box[3]
        total_boxes+=1
        cv2.rectangle(copy,(clean_x1,clean_y1),(clean_x2,clean_y2),(0,255,0),3)
    plt.imshow(copy)
    plt.show()

def main(image_name,model_name):
    test_img = cv2.imread(image_name)
    rcnn(image=test_img,base_model_name=model_name)

main('images\\makeml\\raw\\Faces0.png','models/base_model_v1.h5')