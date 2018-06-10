# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model_architecture.png "Nvidia Model"
[image2]: ./images/model_summary.png "Model Visualization"
[image3]: ./images/loss_graph1.png "Loss Graph"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Nvidia에서 소개한 [모델](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)을 기반으로 네트워크를 구성하였다. 대략적인 구조는 아래와 같다. 

![alt text][image1]

위 모델과 입력 이미지 사이즈가 다른 것을 감안하여 가장 하단의 레이어는 제거하였고 overfitting을 막기 위해 상단 Dense layer 쪽에서는 dropout을 추가하였다. 그리고 마지막 10x1 Dense layer에서 RELU nonlinear activation을 빼주었을 때 더 좋은 성능을 얻을 수 있었다. 

이 프로젝트에서 사용한 최종적인 네트워크 모델은 아래와 같다. 

![alt text][image2]

또한 입력 이미지를 아래와 같이 normalize해주었고, 이미지의 상단 부분은 자동차의 운행 방향을 결정하는데 큰 영향을 미치지 않기 때문에 cropping하여 효율화 및 성능 개선을 꾀하였다. 
```python
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
```

#### 3. Model parameter tuning

예측하려는 값이 categorical 변수가 아닌 numeric 변수이므로 mean squared error loss를 사용하였고, adam optimizer로 30 epoch 정도 학습하였다. 

```python
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
    samples_per_epoch=len(train_samples),
    validation_data=validation_generator, 
    nb_val_samples=len(validation_samples),
    nb_epoch=30, 
    verbose=1)
```

#### 4. Appropriate training data

이 프로젝트에서 성능 향상에 가장 큰 영향을 미친 요소는 위에서 설명한 Nvidia network model을 사용한 것이었다. 하지만 이 네트워크 모델을 도입한것 만으로는 요구사항을 만족할 수 없었는데 해당 모델이 validation accuracy가 높게 나오기는 하지만 막상 시뮬레이션을 돌려보면 직진만 하는 경향을 보였고 급경사에서 도로 바깥으로 이탈하는 경향이 있었다. 이 문제를 해결해 준 것이 바로 data augmentation이었다. 

```python
def generator(samples, batch_size=32, is_training=False):
    ...
        if is_training == True:
          cam_id = randint(0,2)
        else :
          cam_id = 0
    ...
        angle = float(batch_sample[3])
        if cam_id == 1:
          angle += 0.2
        elif cam_id == 2:
          angle -= 0.2
        if is_training == True and random.random() > 0.5:
          image = cv2.flip(image,1)
          angle = -angle
        images.append(image)
        angles.append(angle)
    ...
```

1/2의 확률로 이미지를 flip 해주어 특정 상황에서 자동차가 우회전으로 빙글빙글 도는 현상을 막을 수 있었고, center image와 함께 left, right image를 모두 사용하여 자동차가 도로를 이탈하는 현상을 막을 수 있었다. 

#### 5. Training 

CNN으로 Regression을 학습했기 때문인지는 모르겠지만 학습기를 돌릴 때 상당히 빠르게 오버피팅되는 경향이 있었고, 매우 불안정한 loss 감소 그래프를 관찰할 수 있었다. 

![alt text][image3]

몇번의 시행착오 끝에 시뮬레이션에서 최적의 성능을 보이는 iteration 수(30 epoch)를 찾아냈고, 시뮬레이션 결과 video는 [여기](https://www.youtube.com/watch?v=1okj095apic)에서 참고할 수 있다. 

<div align="left">
  <a href="https://www.youtube.com/watch?v=1okj095apic"><img src="https://img.youtube.com/vi/1okj095apic/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>
