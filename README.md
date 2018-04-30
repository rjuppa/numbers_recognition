# numbers_recognition
Building a small neural network to recognise hand written numbers using TensorFlow.


First, I want to thank the authors of this article that was the inspiration for this project.
<a href="http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/" target="_blank">cv-tricks.com/tensorflow-tutorial</a>
A dataset with handwritten numbers was used from here:
<a href="https://github.com/kensanata/numbers" target="_blank">https://github.com/kensanata/numbers</a>

### Installation  
- It uses Python3 (I recommend to use VirtualEnv)
- [https://opencv.org/](https://opencv.org/) for image manipulation
- [https://www.tensorflow.org](https://www.tensorflow.org)


Make sure that OpenCV is accessible in your Python3 console.
```
$ python3
Python 3.6.0 (default, Dec 24 2016, 08:01:42)
[GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.42.1)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2 as cv
>>> print(cv.__version__)
3.4.1
```

If it works then install [www.tensorflow.org/install](https://www.tensorflow.org/install/)

Finally clone the project:
```
git clone https://github.com/rjuppa/numbers_recognition.git
cd numbers_recognition

pip install -r requirements.txt
```
...

To start training type:
```
python3 main.py
```


----
## How the project works

All training data are stored in a folder `./data/{dataset_name}/{classes}`
When a training process get started it stores its model in folder `./model/`
Once the model is built than it is used for further prediction. The images
to predict are different from training images. They are stored in
a subfolder `./data/{dataset_name}/predict/` The first letter of a filename
is actually `class name` used to check results. Results are printed on a screen
and an html page with images are created.

main.py
```
from classifier import ImageClassifier

model_name = '0025_RO4F'

# train a model
ic = ImageClassifier()
ic.set_classes(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ic.set_model_name(model_name)
ic.set_train_data_path('data/{}'.format(model_name))
ic.train(2000)

# use the model to classify images from a different folder
ic.predict('data/{}/predict/'.format(model_name))
ic.print_results()

# create an html page with a graphic report
ic.print_html_report()
```

----
## Results

I reached positive results.  
Correct prediction of 17 from 20 images. (0001_CH4M)  
Correct prediction of 15 from 20 images. (0025_RO4F)

  <a href="https://rjuppa.github.io/numbers_recognition/report_0001_CH4M.html" title="report_0001_CH4M.html">report_0001_CH4M.html</a>

  <a href="https://rjuppa.github.io/numbers_recognition/report_0025_RO4F.html" title="report_0025_RO4F.html">report_0025_RO4F.html</a>


