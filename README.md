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
![data augment](/Images/data_augment.png)
Next, becasue the datasets will contain 200,000+ images of negative and positive tests, I would like to train a subset of these images simply for time efficiency. It takes about ~4 minutes per epoch with 200,000 images, but around ~30 sec per epoch for a 30,000 image set, so I will go with the latter. Check subset.ipynb for the code to create a subset of the original set. Also, I have opted to use mini-batch training because it offers faster convergence and reduces overfitting as it promotes generalization throughout the data. The batch size I will use is 45 for my subset. 
### The Model
Let's define a few key functions found in each of my models. 
`class ExampleName(nn.Module)` is a class I created that subclasses the nn.Module base class that holds all the necessary methods and attributes needed to create a custom model. Within it, `def __init__ (self)` is my constructor, where my model will be instantiated with the layers I implement. `super().__init__()` calls the constructor of the super class `nn.Module`, inheriting all methods and attributes of `nn.Module`. `nn.Sequential()` is a method that allows you to stack layers in the order in which you implement them for the forward pass. `nn.Conv2d` is a 2 dimensional convolution that takes the output of the previous layer, performs and elementwise dot product with elements following the `kernel_size` hyperparameter. For example, if the `kernel_size` is 3, a unique 3x3 kernel will be applied on the previous output layer's channel by sliding over it, generating the intermediate result. Afterward, the intermediate result are summed and the learned bias is applied to generate the neuron in the next layer. So, to reiterate, if the `in_channels` is three, it takes in Red Green Blue color channels, applies a distinct weight or kernel to each color channel to slideover, and the output of each are summed with the bias to create an output neuron in the next layer. If there are multiple neurons in the next layer, the same process is 
conducted as before, but with another unique kernel to form a different neuron. The `stride` hyperparameter is the step size of the kernel over the convolution. For example, when the stride = 1, the kernel will slide over the input as normal. But, when the stride is 2, the kernel will skip a pixel as it moves. This results in a smaller height and width of the image passing through the convolution, specifically it will be downscaled by a factor of (1/stride). The `padding` hyperparameter is used for preserving the spatial dimensions of the image (height x width). Without padding, convolutions may reduce the dimensions of the image, losing out on data toward the ends of the image. `nn.ReLU()`
