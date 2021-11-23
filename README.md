# Gesture-Detection-and-Depth-Estimation
This is my graduation project.

(1) In this project, I use the YOLOv3 object detection model to detect gesture in RGB image. I trained the model on the self-made gesture dataset to obtain the gesture detection model based on deep learning. Then by testing the model on the test dataset, I found that the model can meet the requirements of real-time gesture detection while maintaining high accuracy.

(2) Then I tried to use the monocular depth estimation algorithm based on depth learning to estimate the depth of gesture object from a single RGB image, including FastDepth algorithm and the improved detection model based on YOLOv3. The FastDepth algorithm is trained and tested on the self-made gesture-depth dataset. Then, by adding a depth vector to output dimensions and modifying the loss function, the function of estimating target depth is added to the YOLOv3 model. Then I trained and tested the modified YOLOv3 model on the same gesture-depth dataset. Finally, the experiment results show that both methods can estimate the depth information of gesture object in RGB image to a certain extent.

Gesture detection:
![image](https://user-images.githubusercontent.com/37933769/142997771-9f96596f-78d2-4de7-a69b-580120975cfa.png)

Depth data:
![image](https://user-images.githubusercontent.com/37933769/142997868-4758cbda-0cb8-4cb7-9bc6-2753912d51d1.png)

Estimate target depth：
![image](https://user-images.githubusercontent.com/37933769/142997909-4316860b-dcc5-4faa-b8b2-9084b3df8057.png)

(3) Also, I developed a simple program with PyOpenGL that can use gesture information to draw simple shapes in three-dimensional space
Try to draw a cube:
![image](https://user-images.githubusercontent.com/37933769/142997955-9722359a-bf7c-40d9-a481-8045ee5e0951.png)

For more information, you can check my final paper.

YOLOv3 model is based on coldlarry's model: https://github.com/coldlarry/YOLOv3-complete-pruning
