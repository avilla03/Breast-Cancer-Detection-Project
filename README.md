# 
Hello! My name is Andrew Villapudua, a student at UC San Diego in La Jolla, California. I am a third year student (at the time of writing this) studying Math-CS. Recently, I have grown an interest for deep learning models which has invoked my study in this field. "How in the world can ChatGPT exist and respond well to such obscure prompts?" I ask myself. In this project, I lay a foundation of growth which I can later apply to future, more complex studies. More specifically, the foundation of convolutional neural networks and their role in image recognition. 

## Resources Used
- [Coursera Neural Networks and Deep Learning Course](https://www.coursera.org/learn/neural-networks-deep-learning)
- [Learn PyTorch for deep learning in a day. Literally.](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=67946s&pp=ygUNbGVhcm4gcHl0b3JjaA%3D%3D) By Daniel Bourke
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- ["Breast Histopathology Images."](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) Dataset published by Paul Mooney on Kaggle

## The Data
I often times question the practicality of many topcis covered in school, so I emphasized the real-world application aspect of this project. So, I opted for a dataset which contians hundreds of thousands of images of breast tissue cells at the microscopic level to classify whether a given image is Invasive Ductal Carcinoma (ICD) positive, or negative. ICD is an invasive breast cancer that forms in the milk ducts, and in later stages inhabits the tissues of the breast. The way this dataset works - or all image recognition datasets for that matter - is it will contain your input data, a [# of color channels, height, width] tensor that represents each individual image. This image will contain a label, in this case, it will classify it as ICD positive or negative depending on the image given. Firstly, I will show the code snippet that takes the data downloaded from the link above on my google drive, into google colab. Please see import_data.ipynb to view the code for this.

The directory "data" should appear now in the files tab with a subdirectory named "med_gallery", with multiple subdirecteries all containing images of ICD positive images with the label 1, and ICD negative images with the label 0.
![import data](/Images/import_data.png)
Great! My next problem is to format the data in a way such that the ImageFolder dataloader can pair well with my data. The general structure of my folders should look like:
- /data
  - /med_gallery
  - /train
    - /0
      - /*.png
      - /*.png
      - ...
    - /1
  - /test
    - /0
    - /1
I was able to accomplish this with the code found in data_structure.ipynb. Now, I would like to augment my data in a way such that it promotes diversity and excites new patterns to be learned in my model. Rather, to likely combat overfitting down the road. So, I will be executing the following transforms: Resize the image to 64 x 64 which is standard. Next, I will flip the data along the horizantal axis with a probability of .6. See the code in data_augment.ipynb. An example of this is as follows. 
![data augment](data_augment.png)
Next, 
