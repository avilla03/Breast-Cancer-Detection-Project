# About Me
My name is Andrew Villapudua, a student at UC San Diego in La Jolla, California. I am a third year student (at the time of writing this) studying Math-CS. I have been running competitively for 4 in both highschool and at the division 1 level, and my hobbies include surfing, skating, and video games. Recently, I have grown an interest for deep learning models which has invoked my study in this field. 
- My [LinkedIn](https://www.linkedin.com/in/andrew-villapudua-998b7425b/)
- My [Instagram](https://www.instagram.com/andrewvillapuada/)
- My Email: andrewvillapudua02@gmail.com
## Resources Used
- [Coursera Neural Networks and Deep Learning Course](https://www.coursera.org/learn/neural-networks-deep-learning)
- [Learn PyTorch for deep learning in a day. Literally.](https://www.youtube.com/watch?v=Z_ikDlimN6A&t=67946s&pp=ygUNbGVhcm4gcHl0b3JjaA%3D%3D) By Daniel Bourke
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- ["Breast Histopathology Images."](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) Dataset published by Paul Mooney on Kaggle
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) by Jay Wang, Robert Turko, Omar Shaikh, Haekyu Park, Nilaksh Das, Fred Hohman, Minsuk Kahng, and Polo Chau
## Introduction
This is an introductory level project that describes convolutional networks and their role in image classification. Let us layout the ground work of neural networks and how they function. Let us take a small sample of data, a 10 image set of 5 cats and 5 dogs for training, and a 4 image set of 2 dogs and 2 cats for testing. The key for the computer to interpret the data we will feed it is through numbers. Each image is comprised of 3 color channels: The colors red, green, and blue which stack on top eachother with numbers in each pixel labeling the varying intensities of R,G, and B. Then, the image also has dimensions, such as height and width. With color channels, height, and width, we can represent any image we'd like. Now that we know how our model will interpret our data, how will it learn it? We will equip our models with parameters, typically named the weight and the bias. The weights and the biases will guide our input tensor through the model until it reaches the end, the output prediction. For example, let us call our input "X," our weight "w," our bias "b," and our output "y." Let us initialize our weight and bias to small random numbers, and compose the function y=wX + b. For reference, here is an image of a simple neural network.

![NN](/Images/SimpleNN.png)

In each neuron in the input layer, there is a randomly assigned weight and bias that produce an output y. Then, all the y's in the input layer are added together to make up the input to one of the neurons in the hidden layer. The neuron the output was sent to in the hidden layer has its own weight and bias parameter, so each neuron in the hidden layer is distinct. A similar process is used to send the outputs to the final layer, where then the prediction on the image is made. Then, we must calculate the loss of the model. The prediction outcome is compared to the true outcome of the image, for example, say the model predicted that the image was a dog, and the true image was a dog, so the loss would be near zero. On the contrary, suppose the model made a wrong prediction, so the loss would be higher. The goal in training models is to lower the loss. After each time the model makes a prediction and the loss is computed is where the model begins to "learn." Recall that we instantiated our weights and biases to random values. We will then compute the gradient (derivative) of each parameter in every layer with respect to the loss function, also known as back propagation, which will then tell us the direction of the steepest increase of the loss. For example, refer to this image. 

![Derivative](/Images/derivative.png)

Naturally, we would like to shift the derivative torward the absolutes of the graph, so we must update the weight and bias by substracting their respective gradient multiplied by the learning rate. The learning rate is a hyperparameter, so its value is up to your discretion. This will then shift the parameters closer to their ideal values. It is clear to see that more data will lead to more shifts in the parameters, demonstrating how important it is to incorporate a large sample size of training and test material. Finally, not only is the function y=wX + b used to interpret the data, but there are many activation outputs to choose from. The most popular in modern-day deep learning models is ReLU, which I will elaborate on in the model section of this project. Activation functions are applied immediately after an output from y=wX + b is computed. In essence, activation functions provide the model with new ways to find patterns in the data, resulting in optimally-suited parameters for our problem.
## The Data
I opted for a dataset which contains hundreds of thousands of images of breast tissue cells at the microscopic level to classify whether a given image is Invasive Ductal Carcinoma (ICD) positive, or negative. ICD is an invasive breast cancer that forms in the milk ducts, and in later stages inhabits the tissues of the breast. The way this dataset works - or all image recognition datasets for that matter - is it will contain your input data, a [# of color channels, height, width] tensor that represents each individual image. This image will contain a label, in this case, it will classify it as ICD positive or negative depending on the image given. Please see import_data.ipynb to view the code for this.
The directory "data" should appear now in the files tab with a subdirectory named "med_gallery", with multiple subdirecteries all containing images of ICD positive images with the label 1, and ICD negative images with the label 0.

![import data](/Images/import_data.png)

Great! My next problem is to format the data in a way such that the ImageFolder dataloader from PyTorch can pair well with my data. The general structure of my folders should look like:
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

I was able to accomplish this with the code found in data_structure.ipynb. Now, I would like to augment my data in a way such that it promotes diversity and excites new patterns to be learned in my model. Rather, to likely combat overfitting in my model down the road which will be explained in the models section of this project. So, I will be executing the following transforms: Resize the image to 64 x 64 which is standard. Next, I will flip the data along the vertical axis, with a probability of .6. See the code in data_augment.ipynb. An example of this is as follows. 
![data augment](/Images/data_augment.png)
Next, because the datasets will contain 200,000+ images of negative and positive tests, I would like to train a subset of these images simply for time efficiency. It takes about ~4 minutes per epoch with 200,000 images, but around ~30 sec per epoch for a 30,000 image set, so I will go with the latter. Check subset.ipynb for the code to create a subset of the original set. Also, I have opted to use mini-batch training because it offers faster convergence and reduces overfitting as it promotes generalization throughout the data. 
### The Model Architecture
To view my models, please see the models folder located in this project repository. Let's define a few key functions found in each of my models. 
`class ExampleName(nn.Module)` is a class I created that subclasses the nn.Module base class that holds all the necessary methods and attributes needed to create a custom model. Within it, `def __init__ (self)` is my constructor, where my model will be instantiated with the layers I implement. `super().__init__()` calls the constructor of the super class `nn.Module`, inheriting all methods and attributes of `nn.Module`. 
`nn.Sequential()` is a method that allows you to stack layers in the order in which you implement them for the forward pass. 
`nn.Conv2d` is a 2 dimensional convolution that takes the output of the previous layer, performs and elementwise dot product with elements following the `kernel_size` hyperparameter. For example, if the `kernel_size` is 3, a unique 3x3 kernel will be applied on the previous output layer's channel by sliding over it, generating the intermediate result. Afterward, the intermediate results are summed and the learned bias is applied to generate the neuron in the next layer. So, to reiterate, if the `in_channels` is three, it takes in Red Green Blue color channels, applies a distinct weight/kernel to each color channel to slideover, and the output of each are summed with the bias to create an output neuron in the next layer. If there are multiple neurons in the next layer, the same process is conducted as before, but with another unique kernel to form a different neuron. The `stride` hyperparameter is the step size of the kernel over the convolution. For example, when the stride = 1, the kernel will slide over the input as normal. But, when the stride is 2, the kernel will skip a pixel as it moves. This results in a smaller height and width of the image passing through the convolution, more precisely, it will be downscaled by a factor of (1/stride). The `padding` hyperparameter is used for preserving the spatial dimensions of the image (height x width). Without padding, convolutions may reduce the dimensions of the image, losing out on data toward the end points and sides. 
`nn.ReLU()` introduces non-linearity to the model, giving the model complex ways to interpret the data to then tune the parameters as needed. Below is an image of this function, as you can see it zeros out the negative values and keeps positives unchanged.

![ReLU Activation Function](/Images/relu.png)

`nn.BatchNorm2d()` normalizes the activation output of each layer it is applied to. It does this by shifting the activation output to have a mean of ~0 and a standard deviation of ~1. This is useful as it introduces more learnable parameters to the model, to adjust the mean and standard deviation after each batch. At the same time, it mitigates the dramatic shifts of learnable parameters nested within the model, leading to faster convergence of the loss function. 
`nn.Dropout()` plays a pivotal role in my model as it mitigates overfitting - the model "memorizes" the training set and thus has a high accuracy on the training set, but fails to generalize these patterns to the test set leading to the loss plateuing with unseen images. This method does this by randomly zeroing a fraction of the input, artificially creating "noise" to the data so it grants the oppurtunity to the model to interpret and generalize the data as opposed to memorizing it. It takes in a hyperparameter `p` which is the probability of the input being zeroed. `nn.Flatten()` simply flattens the 3D tensor passing through the model into a 1D tensor. In this case, since I working with images, the tensor will become [batch_size, color channels, height, width] -> [batch_size, color channels * height * width]. `nn.MaxPool2d()` is commonly used in CNNs to down-sample the data. It reduces the spatial dimension of the data, and has a default stride value of 2 so the height and width are downscaled by 1/2. It takes the size of the kernel, looks at the greatest value within the kernel, and outputs that value.
`nn.Linear()` is a fully connected layer of neurons with the function y=weight*input + bias equipped, then it will typically be followed by a `nn.ReLU()` function to create an activation output. Although not shown in the model, the activation function `torch.sigmoid()` is used as the final output of the function. We use `torch.sigmoid()` as opposed to `torch.softmax()` because this is a binary classification problem, while softmax is used for multiclass classification. The sigmoid layer will act as the final prediction of our model. The function below is the sigmoid function.

![Sigmoid](/Images/sigmoid.png)

It will be applied in the training loop with variable name `y_pred`. It is applied in the training because it is necessary to keep track of logits and predictions. A logit is the output of our entire model before passed through to the sigmoid activation function. This is important because as you will see, the loss function we choose will require that we input logits as our argument, not `y_pred`.
## The Loss Function
There are a plethora of loss functions offered in the pytorch documentation, but there are two that stood out for this problem. Firstly, `nn.BCELoss()` stands for binary cross entropy loss, which essentially acts as measure between the models predicted outputs and the true outputs. Refer below to the equation for BCE. 

![BCE](/Images/BCE.png)

y-hat are the predictions for the ith image sample, and y is the true label for the ith image sample. Secondly, we have `nn.BCEWithLogitsLoss()`. The latter provides a more efficient edge over `nn.BCELoss` because it eliminates the need for a sigmoid layer in our model, as it takes in the logits. In turn, this provides faster training times. Please see loss&optim.ipynb for the implementation.
## The Optimizer
Two optimziers especially stood out to me, `torch.optim.SGD()` which stands for Stochastic Gradient Descent, and `torch.optim.Adam()` which stands for Adaptive Moment Estimation. While SGD is a more traditional optimizer for problems such as binary classification, I chose Adam for a few reasons. Adam is highly adaptive compared to SGD as it automatically updates the learning rate for each parameter when the loss plateaus. Moreover, it contains a momentum-like parameter that adds the previous gradients to help push the gradient when the curves become flatter or it reaches a local minima. Please see loss&optim.ipynb for implementation
## Training Loop and Test Loop
Both loops can be found in the train&test.ipynb file in the project repository. Let's define a few key functions within the loops. An `epoch` is the amount of times the model has passed through the training set and test set. `model.train()` activates the dropout layer, batch normalization, and gradient computation. `model.forward(X).squeeze()` returns the logits of the model when an input is passed through. `.squeeze()` is used for shape congfiguration when multiplying matrices later. `y_pred` are the prediction outcomes of the model, which are passed through the sigmoid function. `loss` is the loss of the current batch, and `train_loss` is the loss summed up over the entire set. `optimizer.zero_grad()` clears gradient accumulation and prepares for the next iteration. `loss.backward` performs backpropogation in the model. It calculates all the model's parameters gradient with respect to the cost function. Then, `optimizer.step()` is called to perform the shift of parameters toward the ideal value. Next, `model.eval()` is called to disable the layers that were activated during training. `with torch.inference_mode():` turns off gradient calculation as it is not needed when testing. 
## Final Results
The results for model 0, a simple non-linear NN were a bit dissapointing. 

![Model0](/Images/model0.png)

As you can see, it seems that the model is underfitting to the data provided. This occurs when the model cannot find the correct fit to the parameters for a number of reasons such as not enough hidden units, not enough layers, too little data, etc. Next, let us examine the model1, this time with with a convolution added as well as a fully connected layer. 

![Model1](/Images/model1.png)

It seems that the training loss goes down steadily, but the test loss fails to converge. Although, it seems to still underfit as the training loss seems to slow down after the first few epochs. Let us try a third and final model to fit the data.

![Model2](/Images/model2.png)

This mdoel has been fitted with 4 convolutions, 4 pooling layers, 2 dropout layers, and a fully connected layer. It seems that the test loss was extremely volatile, but it still received an 85% accuracy on the unseen images. Although the accuracy is not as high as I would have liked, I am left with the knowledge to build deeper, more complex models in the future for other projects which I am excited for. Thank you!
