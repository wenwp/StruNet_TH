# StruNet_TH
Structural nonlinear seismic time-history response prediction of urban-scale reinforced concrete frames based on deep learning

Efficiently predicting the seismic response of urban building clusters is essential for preemptively identifying potential seismic hazards prior to an earthquake and optimizing resource allocation post-event. However, complete information of buildings at a city scale is generally un-accessible or non-existent. Existing methods struggle to reconcile low information demands, high computational accuracy, and computational efficiency. 
We propose a fast prediction method for structural seismic time-history responses that combines deep learning methods with easy-getting structural parameters at an urban scale. An end-to-end network  (named StruNet_TH) with adaptive multilevel fusion output is designed, which incorporates the autoencoder concept for predicting the structural seismic time-history responses based on ground motions records and five easy-getting structural parameters. The models are compared and optimized considering the training hyperparameters and network architecture, resulting in an optimized model with low complexity that provides valuable reference values for structural seismic response. Besides, the proposed model is applied to four actual buildings with different construction time, occupancy types, and floor sizes, demonstrating its good prediction performance and significant computational advantages comparing to the universally used MDOF method.

For more information, please refer to the following:
Zhang C, Wen W, Zhai C, Jia J, Zhou B. Structural nonlinear seismic time-history response prediction of urban-scale reinforced concrete frames based on deep learning. Engineering Structures. 2024;317:118702.
https://doi.org/10.1016/j.engstruct.2024.118702


In other related articles, we also proposed an end-to-end model based on convolutional networks for predicting structural seismic amplitude response (StruNet) in 2022. For more details, please refer to the paper:
Wen W, Zhang C, Zhai C. Rapid seismic response prediction of RC frames based on deep learning and limited building information. Engineering Structures. 2022;267:114638.
https://doi.org/10.1016/j.engstruct.2022.114638



The model proposed in this paper can simultaneously consider the diversity of structural parameters and ground motions. The trained model is applicable to low-rise and mid-rise RC frame structures with different seismic design intensities, floor heights, and plan dimensions. This is the most significant difference compared to the existing studies.


This repository contains the following content:
1. Optimized model 
   1.1 Introduction to model input and output
   1.2 Saved model file (.h5) 
   1.3 Code for reusing the model

2. Four real RC buildings
   2.1 Details of these buildings and the detailed finite element modeling in OpenSees
   2.2 Selected ground motions
   2.3 Calculated seismic responses using Opensees
   2.4 The input and output of Optimized model 



The related data, models, and codes used in this study will be available within the next couple of days. You are welcome to "Watch" and "Star".
