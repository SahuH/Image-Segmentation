# Image-Segmentation

## Approach

I've built a Custom UNet architecture - a modified U-Net architecture that incorporates residual units in each encoder and decoder block of the architecture. I've also introduced new custom loss - an aggregation of binary crossentropy loss, dice coefficient loss, and inverse dice coefficient loss. With careful selection of data augmentation techniques, my Custom U-Net achieved the accuracy of 91.89% and IOU of 88.36%. Using Custom U-Net, I segmented regions of interest from CXR images and then employed a CNN-based classifier to perform COVID-19 detection on these segmented images, achieving an accuracy of 94.13%.

![image](https://github.com/SahuH/Image-Segmentation/assets/28728457/10082c52-e37e-4abf-bb90-7c94f9dff427)


## Results

<img width="460" alt="image" src="https://github.com/SahuH/Image-Segmentation/assets/28728457/ca8f4918-557f-4550-861e-89e861d64051">

<img width="406" alt="image" src="https://github.com/SahuH/Image-Segmentation/assets/28728457/2e73cd5f-bc94-4630-8128-831379ec1884">





