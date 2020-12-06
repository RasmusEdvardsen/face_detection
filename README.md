# Face Detection

## TODO
* sliding window
* load model and predict
* support multiple directories for inputs
* data augmentation
* look at image size transformations (higher or lower www x hhh)
* support black/white images

## Usages
* https://datasetsearch.research.google.com/
* https://github.com/zarif101/rcnn_keras_license_plate

### Tensorboard
python C:\Users\rasmus.edvardsen\AppData\Roaming\Python\Python38\site-packages\tensorboard\main.py --logdir=logs

## Citations
* **selective** search segmentation: (as opposed to **sliding windows**)
http://vision.stanford.edu/teaching/cs231b_spring1415/slides/ssearch_schuyler.pdf

## Dataset Origin
* mnist
* (found in search on) https://datasetsearch.research.google.com/search?query=object%20detection%20face&docid=AIe%2FEvYJ5YTrhEt9AAAAAA%3D%3D
* (actual location) https://makeml.app/datasets/faces
* http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
* https://datasetsearch.research.google.com/search?query=caltech&docid=OObpJrWLGbVqQ1wRAAAAAA%3D%3D