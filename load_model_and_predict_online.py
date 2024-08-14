'''
Local Hardware:
    NVIDIA RTX 3060

requirements:
    python==3.7.13
    tensorflow==2.7.0
    numpy==1.21.5
    pandas==1.3.5
    
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model


#%% model component class

from tensorflow import keras
from tensorflow.python.ops import standard_ops

class AdjustableFusionLayer_4ValueWeight_5Block(keras.layers.Layer):
    def __init__(self, init_value=0.25, **kwargs):
        self.init_value = init_value
        super(AdjustableFusionLayer_4ValueWeight_5Block, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.w1 = tf.Variable(initial_value=tf.constant([self.init_value], shape=[input_shape[1][2]], dtype=tf.float32), trainable=True, name='kernel1')
        self.w2 = tf.Variable(initial_value=tf.constant([self.init_value], shape=[input_shape[1][2]], dtype=tf.float32), trainable=True, name='kernel2')
        self.w3 = tf.Variable(initial_value=tf.constant([self.init_value], shape=[input_shape[1][2]], dtype=tf.float32), trainable=True, name='kernel3')
        self.w4 = tf.Variable(initial_value=tf.constant([self.init_value], shape=[input_shape[1][2]], dtype=tf.float32), trainable=True, name='kernel4')
        self.w5 = tf.Variable(initial_value=tf.constant([self.init_value], shape=[input_shape[1][2]], dtype=tf.float32), trainable=True, name='kernel5')
        super(AdjustableFusionLayer_4ValueWeight_5Block, self).build(input_shape) 
    
    def call(self, inputs):
        x1, x2, x3, x4, x5 = inputs
        summary = standard_ops.tensordot(x1, self.w1, axes=1) + standard_ops.tensordot(x2, self.w2, axes=1) + standard_ops.tensordot(x3, self.w3, axes=1) + standard_ops.tensordot(x4, self.w4, axes=1) + standard_ops.tensordot(x5, self.w5, axes=1)
        return tf.expand_dims(summary, axis=-1) 
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'init_value': self.init_value,
        })
        return config
    
#%% basic functions

def time_history_curve(y_pred_ref, y_pred, label, categ, saveDir=[], count=20):
    num_range = np.random.randint(0,len(y_pred),count)
    if len(y_pred_ref.shape) == 2:
        y_pred_ref = y_pred_ref[:,:,np.newaxis]
        y_pred = y_pred[:,:,np.newaxis]
    elif len(y_pred_ref.shape) != 3:
        print('The dimension of inputs is not 2D or 3D')
    import matplotlib.pyplot as plt
    for ii in num_range:
        plt.figure()
        plt.plot(y_pred_ref[ii, :, 0], label='OpenSees')
        plt.plot(y_pred[ii, :, 0], label='Prediction')
        plt.xlabel('time series (0.02s)')
        plt.ylabel(str(label) + ' (m/s2)')
        label2 = 'THC_' + label + ': case_' + str(ii)
        plt.title(label2)
        plt.legend()
        if len(saveDir) > 0: 
            save_path = saveDir + '/TimeHistoryPerformance_' + label + '_case_' + str(ii) + '.png' 
            plt.savefig(save_path) # , bbox_inches='tight'
        plt.close()
    return 


def predict_on_multiVals_TH(model,val_ag,val_vf,val_absAcc,val_absAcc_spectra,save_dir,certain_array_val,label_val='val_absAcc'): 
    print("start to evaluate on validation dataset...")
    val_absAcc_pred = model.predict([val_ag,val_vf])
    time_history_curve(val_absAcc, val_absAcc_pred, label_val, 'meaningless', save_dir, 20)
    return


def load_and_evaluate_THandSpectra(CAE_path,epochs=300,model=None,pred_THandSpectra=False,train_usingVal=True):
    global Fcases_ag,Fcases_vf,Fcases_absAcc,Fcases_absAcc_spectra
    
    if model == None:
        model = load_model("model/" + CAE_path) # 300个epoch
    else:
        model = model
        
    save_dir = "results/" + CAE_path[:-3]
    isExists = os.path.exists(save_dir)
    if not isExists:
      os.makedirs(save_dir) 
    # from tensorflow.python.keras.utils.vis_utils import plot_model
    plot_model(model, to_file=save_dir+'/model.png', show_shapes=True) 
    
    isExists_hist = os.path.exists("model/" + CAE_path[:-3] + '.csv') 
    if isExists_hist == True:
        hist = pd.read_csv("model/" + CAE_path[:-3] + '.csv') # ,header='infer'
        plt.figure()
        plt.plot(hist["loss"],label='mae')
        plt.plot(hist["mse"],label='mse')
        if train_usingVal == True:
            plt.plot(hist["val_loss"],label='val_loss')
            plt.plot(hist["val_mse"],label='val_mse')
        plt.title('loss in training process')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(save_dir+'/lossCurve.png')
        plt.show()
        
        loss_init = hist["loss"][0]
        loss_end = hist["loss"][epochs-1]
        loss_incicator = [str(round(loss_end, 4)) + ' from ' + str(round(loss_init, 4)) + ' ' + str((loss_init-loss_end)/loss_init)]
        np.savetxt(save_dir+'/output_index_loss.txt',loss_incicator,fmt = '%s')
    else:
        print('')
        print("###### lack loss-history file ######")
        print('')
    np.savetxt(save_dir+'/model_total_param_'+str(model.count_params())+'.txt',[])
    
    if not pred_THandSpectra:
        predict_on_multiVals = predict_on_multiVals_TH
        
    print("start to evaluate on 4 cases...")
    label_Fcases = 'Fcases_absAcc' 
    certain_array_Fcases = sorted(np.random.randint(0,len(Fcases_absAcc),20))
    # certain_array_Fcases = np.array([9, 15, 52, 54, 72, 81, 104, 133, 244, 294, 304, 323, 348, 353, 392, 400, 409, 410, 411, 413])
    predict_on_multiVals(model,Fcases_ag,Fcases_vf,Fcases_absAcc,Fcases_absAcc_spectra,save_dir,certain_array_Fcases,label_Fcases)
    
    print(CAE_path,' IS DONE')
    
    return 
    
    
#%% main
    
# 4cases 
Fcases_ag = np.load("data/input_GMrecorder_420_3000_1.npy") # (200,3000)
Fcases_ag = Fcases_ag.astype('float32')   

Fcases_vf = np.load("data/input_processed_struInfo_420_15.npy")
Fcases_vf = Fcases_vf.astype('float32')

Fcases_absAcc = np.load("data/label_AbsAcc_420_3000_1.npy")
Fcases_absAcc = Fcases_absAcc.astype('float32')

Fcases_absAcc_spectra = np.load("data/label_AbsAcc_spectra_420_3000_1.npy")
Fcases_absAcc_spectra = Fcases_absAcc_spectra.astype('float32')



# load model and predict
CAE_path = 'cae_CR10_concatenate_5block_3kernelSize_1layers_4ValueWeightInit1d0_lr001_finalModel.h5'

model = load_model("model/" + CAE_path,custom_objects={'AdjustableFusionLayer_4ValueWeight_5Block':AdjustableFusionLayer_4ValueWeight_5Block})

load_and_evaluate_THandSpectra(CAE_path,300,model)

        