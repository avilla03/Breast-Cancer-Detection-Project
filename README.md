# 
Hello! My name is Andrew Villapudua, a student at UC San Diego in La Jolla, California. I am a third year student (at the time of writing this) studying Math-CS. Recently, I have grown an interest for deep learning models which has invoked my study in this field. "How in the world can ChatGPT exist and respond well to such obscure prompts?" I ask myself. In this project, I lay a foundation of growth which I can later apply to future, more complex studies. More specifically, the foundation of convolutional neural networks and their role in image recognition. 

## Resources Used
- [Coursera Neural Networks and Deep Learning Course](https://www.coursera.org/learn/neural-networks-deep-learning)
- [Learn PyTorch for deep learning in a day. Literally.](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=67946s&pp=ygUNbGVhcm4gcHl0b3JjaA%3D%3D) By Daniel Bourke
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- ["Breast Histopathology Images."](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) Dataset published by Paul Mooney on Kaggle

## The Data
I often times question the practicality of many topcis covered in school, so I emphasized the real-world application aspect of this project. So, I opted for a dataset which contians hundreds of thousands of images of breast tissue cells at the microscopic level to classify whether a given image is Invasive Ductal Carcinoma (ICD) positive, or negative. ICD is an invasive breast cancer that forms in the milk ducts, and in later stages inhabits the tissues of the breast. The way this dataset works - or all image recognition datasets for that matter - is it will contain your input data, a [# of color channels, height, width] tensor that represents each individual image. This image will contain a label, in this case, it will classify it as IDC positive or negative depending on the image given. Firstly, I will show the code snippet that takes the data downloaded from the link above on my google drive, into google colab.
```
import requests # Get info from HTTP link
import zipfile # Import data in form of a zipfile
from pathlib import Path # Store path of data
data_path = Path("data/")
images_path = data_path / "med_gallery"
if images_path.is_dir():
  print(f"{images_path} directory already exists, skipping download")
else:
  print(f"{images_path} does not exist, creating one...")
  images_path.mkdir(parents=True, exist_ok=True) # make directory and create parent directories
                                               # if needed. It is ok if this dir exists already
# specifying the zip file name from google drive & extraction paths
zip_file_path = "/content/drive/MyDrive/archive.zip"
extraction_path = images_path
with zipfile.ZipFile(zip_file_path, 'r') as zip:
  print('Extracting all the files now...')
  zip.extractall(extraction_path)
  print('Done!')
```
The directory "data" should appear now in the files tab with a subdirectory named "med_gallery", with multiple subdirecteries all containing images of ICD positive images with the label 1, and ICD negative images with the label 0.
![import data](Convolutional-NN-Project/Images
/import_data.png)
