# StruNet_TH
Structural nonlinear seismic time-history response prediction of urban-scale reinforced concrete frames based on deep learning

Efficiently predicting the seismic response of urban building clusters is essential for preemptively identifying potential seismic hazards prior to an earthquake and optimizing resource allocation post-event. However, complete information of buildings at a city scale is generally un-accessible or non-existent. Existing methods struggle to reconcile low information demands, high computational accuracy, and computational efficiency. 
We propose a fast prediction method for structural seismic time-history responses that combines deep learning methods with easy-getting structural parameters at an urban scale. **An end-to-end network  (named StruNet_TH) with adaptive multilevel fusion output is designed, which incorporates the autoencoder concept for predicting the structural seismic time-history responses based on ground motions records and five easy-getting structural parameters.** The models are compared and optimized considering the training hyperparameters and network architecture, resulting in an optimized model with low complexity that provides valuable reference values for structural seismic response. Besides, the proposed model is applied to four actual buildings with different construction time, occupancy types, and floor sizes, demonstrating its good prediction performance and significant computational advantages comparing to the universally used MDOF method.

For more information, please refer to the following:
* Zhang C, Wen W, Zhai C, Jia J, Zhou B. [Structural nonlinear seismic time-history response prediction of urban-scale reinforced concrete frames based on deep learning](https://doi.org/10.1016/j.engstruct.2024.118702). Engineering Structures. 2024;317:118702.

Besides, we proposed an end-to-end model based on convolutional networks for predicting structural seismic amplitude response (named StruNet) in 2022. For more details, please refer to the paper:
* Wen W, Zhang C, Zhai C. [Rapid seismic response prediction of RC frames based on deep learning and limited building information](https://doi.org/10.1016/j.engstruct.2022.114638). Engineering Structures. 2022;267:114638.


## This repository for StruNet_TH
The model proposed in this paper can simultaneously consider the diversity of structural parameters and ground motions. The trained model is applicable to low-rise and mid-rise RC frame structures with different seismic design intensities, floor heights, and plan dimensions. This is the most significant difference compared to the existing studies.

This repository contains the following content:
<pre>
1. The final model and loss curves from the paper are provided in 'model' folder.

2. Four real RC buildings and 20 ground motions in the Case Study,
   2.1 Calculated seismic responses using Opensee (OpenSees results of 4cases.rar)
   2.2 The input and output of Optimized model ('data' and 'results')
</pre>


## Requirements
    python==3.7.13    
    tensorflow==2.7.0



## Citation
<pre>
@article{zhang2024strunetth,  
         title={Structural nonlinear seismic time-history response prediction of urban-scale reinforced concrete frames based on deep learning},  
         author={Zhang, Chenyu and Wen, Weiping and Zhai, Changhai and Jia, Jun and Zhou, Bochang}, 
         journal={Engineering Structures},  
         volume={317},  
         pages={118702},  
         year={2024},  
         publisher={Elsevier}  
         }

@article{wen2022strunet,  
         title={Rapid seismic response prediction of RC frames based on deep learning and limited building information},  
         author={Wen, Weiping and Zhang, Chenyu and Zhai, Changhai}, 
         journal={Engineering Structures},  
         volume={267},  
         pages={114638},  
         year={2022},  
         publisher={Elsevier}  
         }

</pre>
