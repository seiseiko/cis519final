C:\Users\seise\PycharmProjects\cis519final\venv2\Scripts\python.exe "C:\Program Files\JetBrains\PyCharm 2021.2.2\plugins\python\helpers\pydev\pydevconsole.py" --mode=client --port=64822
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\seise\\PycharmProjects\\cis519final', 'C:/Users/seise/PycharmProjects/cis519final'])
PyDev console: starting.
Python 3.10.0 (tags/v3.10.0:b494f59, Oct  4 2021, 19:00:18) [MSC v.1929 64 bit (AMD64)] on win32
runfile('C:/Users/seise/PycharmProjects/cis519final/mobilenet.py', wdir='C:/Users/seise/PycharmProjects/cis519final')
Found 12121 images belonging to 3 classes.
Found 1515 images belonging to 3 classes.
WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
2022-12-06 23:13:50.454298: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-12-06 23:13:50.843216: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9436 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3080 Ti, pci bus id: 0000:01:00.0, compute capability: 8.6
Model: "mobilenetv2_1.00_224"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 256, 256, 3  0           []                               
                                )]                                                                
                                                                                                  
 Conv1 (Conv2D)                 (None, 128, 128, 32  864         ['input_1[0][0]']                
                                )                                                                 
                                                                                                  
 bn_Conv1 (BatchNormalization)  (None, 128, 128, 32  128         ['Conv1[0][0]']                  
                                )                                                                 
                                                                                                  
 Conv1_relu (ReLU)              (None, 128, 128, 32  0           ['bn_Conv1[0][0]']               
                                )                                                                 
                                                                                                  
 expanded_conv_depthwise (Depth  (None, 128, 128, 32  288        ['Conv1_relu[0][0]']             
 wiseConv2D)                    )                                                                 
                                                                                                  
 expanded_conv_depthwise_BN (Ba  (None, 128, 128, 32  128        ['expanded_conv_depthwise[0][0]']
 tchNormalization)              )                                                                 
                                                                                                  
 expanded_conv_depthwise_relu (  (None, 128, 128, 32  0          ['expanded_conv_depthwise_BN[0][0
 ReLU)                          )                                ]']                              
                                                                                                  
 expanded_conv_project (Conv2D)  (None, 128, 128, 16  512        ['expanded_conv_depthwise_relu[0]
                                )                                [0]']                            
                                                                                                  
 expanded_conv_project_BN (Batc  (None, 128, 128, 16  64         ['expanded_conv_project[0][0]']  
 hNormalization)                )                                                                 
                                                                                                  
 block_1_expand (Conv2D)        (None, 128, 128, 96  1536        ['expanded_conv_project_BN[0][0]'
                                )                                ]                                
                                                                                                  
 block_1_expand_BN (BatchNormal  (None, 128, 128, 96  384        ['block_1_expand[0][0]']         
 ization)                       )                                                                 
                                                                                                  
 block_1_expand_relu (ReLU)     (None, 128, 128, 96  0           ['block_1_expand_BN[0][0]']      
                                )                                                                 
                                                                                                  
 block_1_pad (ZeroPadding2D)    (None, 129, 129, 96  0           ['block_1_expand_relu[0][0]']    
                                )                                                                 
                                                                                                  
 block_1_depthwise (DepthwiseCo  (None, 64, 64, 96)  864         ['block_1_pad[0][0]']            
 nv2D)                                                                                            
                                                                                                  
 block_1_depthwise_BN (BatchNor  (None, 64, 64, 96)  384         ['block_1_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_1_depthwise_relu (ReLU)  (None, 64, 64, 96)   0           ['block_1_depthwise_BN[0][0]']   
                                                                                                  
 block_1_project (Conv2D)       (None, 64, 64, 24)   2304        ['block_1_depthwise_relu[0][0]'] 
                                                                                                  
 block_1_project_BN (BatchNorma  (None, 64, 64, 24)  96          ['block_1_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_2_expand (Conv2D)        (None, 64, 64, 144)  3456        ['block_1_project_BN[0][0]']     
                                                                                                  
 block_2_expand_BN (BatchNormal  (None, 64, 64, 144)  576        ['block_2_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_2_expand_relu (ReLU)     (None, 64, 64, 144)  0           ['block_2_expand_BN[0][0]']      
                                                                                                  
 block_2_depthwise (DepthwiseCo  (None, 64, 64, 144)  1296       ['block_2_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_2_depthwise_BN (BatchNor  (None, 64, 64, 144)  576        ['block_2_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_2_depthwise_relu (ReLU)  (None, 64, 64, 144)  0           ['block_2_depthwise_BN[0][0]']   
                                                                                                  
 block_2_project (Conv2D)       (None, 64, 64, 24)   3456        ['block_2_depthwise_relu[0][0]'] 
                                                                                                  
 block_2_project_BN (BatchNorma  (None, 64, 64, 24)  96          ['block_2_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_2_add (Add)              (None, 64, 64, 24)   0           ['block_1_project_BN[0][0]',     
                                                                  'block_2_project_BN[0][0]']     
                                                                                                  
 block_3_expand (Conv2D)        (None, 64, 64, 144)  3456        ['block_2_add[0][0]']            
                                                                                                  
 block_3_expand_BN (BatchNormal  (None, 64, 64, 144)  576        ['block_3_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_3_expand_relu (ReLU)     (None, 64, 64, 144)  0           ['block_3_expand_BN[0][0]']      
                                                                                                  
 block_3_pad (ZeroPadding2D)    (None, 65, 65, 144)  0           ['block_3_expand_relu[0][0]']    
                                                                                                  
 block_3_depthwise (DepthwiseCo  (None, 32, 32, 144)  1296       ['block_3_pad[0][0]']            
 nv2D)                                                                                            
                                                                                                  
 block_3_depthwise_BN (BatchNor  (None, 32, 32, 144)  576        ['block_3_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_3_depthwise_relu (ReLU)  (None, 32, 32, 144)  0           ['block_3_depthwise_BN[0][0]']   
                                                                                                  
 block_3_project (Conv2D)       (None, 32, 32, 32)   4608        ['block_3_depthwise_relu[0][0]'] 
                                                                                                  
 block_3_project_BN (BatchNorma  (None, 32, 32, 32)  128         ['block_3_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_4_expand (Conv2D)        (None, 32, 32, 192)  6144        ['block_3_project_BN[0][0]']     
                                                                                                  
 block_4_expand_BN (BatchNormal  (None, 32, 32, 192)  768        ['block_4_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_4_expand_relu (ReLU)     (None, 32, 32, 192)  0           ['block_4_expand_BN[0][0]']      
                                                                                                  
 block_4_depthwise (DepthwiseCo  (None, 32, 32, 192)  1728       ['block_4_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_4_depthwise_BN (BatchNor  (None, 32, 32, 192)  768        ['block_4_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_4_depthwise_relu (ReLU)  (None, 32, 32, 192)  0           ['block_4_depthwise_BN[0][0]']   
                                                                                                  
 block_4_project (Conv2D)       (None, 32, 32, 32)   6144        ['block_4_depthwise_relu[0][0]'] 
                                                                                                  
 block_4_project_BN (BatchNorma  (None, 32, 32, 32)  128         ['block_4_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_4_add (Add)              (None, 32, 32, 32)   0           ['block_3_project_BN[0][0]',     
                                                                  'block_4_project_BN[0][0]']     
                                                                                                  
 block_5_expand (Conv2D)        (None, 32, 32, 192)  6144        ['block_4_add[0][0]']            
                                                                                                  
 block_5_expand_BN (BatchNormal  (None, 32, 32, 192)  768        ['block_5_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_5_expand_relu (ReLU)     (None, 32, 32, 192)  0           ['block_5_expand_BN[0][0]']      
                                                                                                  
 block_5_depthwise (DepthwiseCo  (None, 32, 32, 192)  1728       ['block_5_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_5_depthwise_BN (BatchNor  (None, 32, 32, 192)  768        ['block_5_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_5_depthwise_relu (ReLU)  (None, 32, 32, 192)  0           ['block_5_depthwise_BN[0][0]']   
                                                                                                  
 block_5_project (Conv2D)       (None, 32, 32, 32)   6144        ['block_5_depthwise_relu[0][0]'] 
                                                                                                  
 block_5_project_BN (BatchNorma  (None, 32, 32, 32)  128         ['block_5_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_5_add (Add)              (None, 32, 32, 32)   0           ['block_4_add[0][0]',            
                                                                  'block_5_project_BN[0][0]']     
                                                                                                  
 block_6_expand (Conv2D)        (None, 32, 32, 192)  6144        ['block_5_add[0][0]']            
                                                                                                  
 block_6_expand_BN (BatchNormal  (None, 32, 32, 192)  768        ['block_6_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_6_expand_relu (ReLU)     (None, 32, 32, 192)  0           ['block_6_expand_BN[0][0]']      
                                                                                                  
 block_6_pad (ZeroPadding2D)    (None, 33, 33, 192)  0           ['block_6_expand_relu[0][0]']    
                                                                                                  
 block_6_depthwise (DepthwiseCo  (None, 16, 16, 192)  1728       ['block_6_pad[0][0]']            
 nv2D)                                                                                            
                                                                                                  
 block_6_depthwise_BN (BatchNor  (None, 16, 16, 192)  768        ['block_6_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_6_depthwise_relu (ReLU)  (None, 16, 16, 192)  0           ['block_6_depthwise_BN[0][0]']   
                                                                                                  
 block_6_project (Conv2D)       (None, 16, 16, 64)   12288       ['block_6_depthwise_relu[0][0]'] 
                                                                                                  
 block_6_project_BN (BatchNorma  (None, 16, 16, 64)  256         ['block_6_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_7_expand (Conv2D)        (None, 16, 16, 384)  24576       ['block_6_project_BN[0][0]']     
                                                                                                  
 block_7_expand_BN (BatchNormal  (None, 16, 16, 384)  1536       ['block_7_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_7_expand_relu (ReLU)     (None, 16, 16, 384)  0           ['block_7_expand_BN[0][0]']      
                                                                                                  
 block_7_depthwise (DepthwiseCo  (None, 16, 16, 384)  3456       ['block_7_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_7_depthwise_BN (BatchNor  (None, 16, 16, 384)  1536       ['block_7_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_7_depthwise_relu (ReLU)  (None, 16, 16, 384)  0           ['block_7_depthwise_BN[0][0]']   
                                                                                                  
 block_7_project (Conv2D)       (None, 16, 16, 64)   24576       ['block_7_depthwise_relu[0][0]'] 
                                                                                                  
 block_7_project_BN (BatchNorma  (None, 16, 16, 64)  256         ['block_7_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_7_add (Add)              (None, 16, 16, 64)   0           ['block_6_project_BN[0][0]',     
                                                                  'block_7_project_BN[0][0]']     
                                                                                                  
 block_8_expand (Conv2D)        (None, 16, 16, 384)  24576       ['block_7_add[0][0]']            
                                                                                                  
 block_8_expand_BN (BatchNormal  (None, 16, 16, 384)  1536       ['block_8_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_8_expand_relu (ReLU)     (None, 16, 16, 384)  0           ['block_8_expand_BN[0][0]']      
                                                                                                  
 block_8_depthwise (DepthwiseCo  (None, 16, 16, 384)  3456       ['block_8_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_8_depthwise_BN (BatchNor  (None, 16, 16, 384)  1536       ['block_8_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_8_depthwise_relu (ReLU)  (None, 16, 16, 384)  0           ['block_8_depthwise_BN[0][0]']   
                                                                                                  
 block_8_project (Conv2D)       (None, 16, 16, 64)   24576       ['block_8_depthwise_relu[0][0]'] 
                                                                                                  
 block_8_project_BN (BatchNorma  (None, 16, 16, 64)  256         ['block_8_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_8_add (Add)              (None, 16, 16, 64)   0           ['block_7_add[0][0]',            
                                                                  'block_8_project_BN[0][0]']     
                                                                                                  
 block_9_expand (Conv2D)        (None, 16, 16, 384)  24576       ['block_8_add[0][0]']            
                                                                                                  
 block_9_expand_BN (BatchNormal  (None, 16, 16, 384)  1536       ['block_9_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_9_expand_relu (ReLU)     (None, 16, 16, 384)  0           ['block_9_expand_BN[0][0]']      
                                                                                                  
 block_9_depthwise (DepthwiseCo  (None, 16, 16, 384)  3456       ['block_9_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_9_depthwise_BN (BatchNor  (None, 16, 16, 384)  1536       ['block_9_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_9_depthwise_relu (ReLU)  (None, 16, 16, 384)  0           ['block_9_depthwise_BN[0][0]']   
                                                                                                  
 block_9_project (Conv2D)       (None, 16, 16, 64)   24576       ['block_9_depthwise_relu[0][0]'] 
                                                                                                  
 block_9_project_BN (BatchNorma  (None, 16, 16, 64)  256         ['block_9_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_9_add (Add)              (None, 16, 16, 64)   0           ['block_8_add[0][0]',            
                                                                  'block_9_project_BN[0][0]']     
                                                                                                  
 block_10_expand (Conv2D)       (None, 16, 16, 384)  24576       ['block_9_add[0][0]']            
                                                                                                  
 block_10_expand_BN (BatchNorma  (None, 16, 16, 384)  1536       ['block_10_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_10_expand_relu (ReLU)    (None, 16, 16, 384)  0           ['block_10_expand_BN[0][0]']     
                                                                                                  
 block_10_depthwise (DepthwiseC  (None, 16, 16, 384)  3456       ['block_10_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_10_depthwise_BN (BatchNo  (None, 16, 16, 384)  1536       ['block_10_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_10_depthwise_relu (ReLU)  (None, 16, 16, 384)  0          ['block_10_depthwise_BN[0][0]']  
                                                                                                  
 block_10_project (Conv2D)      (None, 16, 16, 96)   36864       ['block_10_depthwise_relu[0][0]']
                                                                                                  
 block_10_project_BN (BatchNorm  (None, 16, 16, 96)  384         ['block_10_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_11_expand (Conv2D)       (None, 16, 16, 576)  55296       ['block_10_project_BN[0][0]']    
                                                                                                  
 block_11_expand_BN (BatchNorma  (None, 16, 16, 576)  2304       ['block_11_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_11_expand_relu (ReLU)    (None, 16, 16, 576)  0           ['block_11_expand_BN[0][0]']     
                                                                                                  
 block_11_depthwise (DepthwiseC  (None, 16, 16, 576)  5184       ['block_11_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_11_depthwise_BN (BatchNo  (None, 16, 16, 576)  2304       ['block_11_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_11_depthwise_relu (ReLU)  (None, 16, 16, 576)  0          ['block_11_depthwise_BN[0][0]']  
                                                                                                  
 block_11_project (Conv2D)      (None, 16, 16, 96)   55296       ['block_11_depthwise_relu[0][0]']
                                                                                                  
 block_11_project_BN (BatchNorm  (None, 16, 16, 96)  384         ['block_11_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_11_add (Add)             (None, 16, 16, 96)   0           ['block_10_project_BN[0][0]',    
                                                                  'block_11_project_BN[0][0]']    
                                                                                                  
 block_12_expand (Conv2D)       (None, 16, 16, 576)  55296       ['block_11_add[0][0]']           
                                                                                                  
 block_12_expand_BN (BatchNorma  (None, 16, 16, 576)  2304       ['block_12_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_12_expand_relu (ReLU)    (None, 16, 16, 576)  0           ['block_12_expand_BN[0][0]']     
                                                                                                  
 block_12_depthwise (DepthwiseC  (None, 16, 16, 576)  5184       ['block_12_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_12_depthwise_BN (BatchNo  (None, 16, 16, 576)  2304       ['block_12_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_12_depthwise_relu (ReLU)  (None, 16, 16, 576)  0          ['block_12_depthwise_BN[0][0]']  
                                                                                                  
 block_12_project (Conv2D)      (None, 16, 16, 96)   55296       ['block_12_depthwise_relu[0][0]']
                                                                                                  
 block_12_project_BN (BatchNorm  (None, 16, 16, 96)  384         ['block_12_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_12_add (Add)             (None, 16, 16, 96)   0           ['block_11_add[0][0]',           
                                                                  'block_12_project_BN[0][0]']    
                                                                                                  
 block_13_expand (Conv2D)       (None, 16, 16, 576)  55296       ['block_12_add[0][0]']           
                                                                                                  
 block_13_expand_BN (BatchNorma  (None, 16, 16, 576)  2304       ['block_13_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_13_expand_relu (ReLU)    (None, 16, 16, 576)  0           ['block_13_expand_BN[0][0]']     
                                                                                                  
 block_13_pad (ZeroPadding2D)   (None, 17, 17, 576)  0           ['block_13_expand_relu[0][0]']   
                                                                                                  
 block_13_depthwise (DepthwiseC  (None, 8, 8, 576)   5184        ['block_13_pad[0][0]']           
 onv2D)                                                                                           
                                                                                                  
 block_13_depthwise_BN (BatchNo  (None, 8, 8, 576)   2304        ['block_13_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_13_depthwise_relu (ReLU)  (None, 8, 8, 576)   0           ['block_13_depthwise_BN[0][0]']  
                                                                                                  
 block_13_project (Conv2D)      (None, 8, 8, 160)    92160       ['block_13_depthwise_relu[0][0]']
                                                                                                  
 block_13_project_BN (BatchNorm  (None, 8, 8, 160)   640         ['block_13_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_14_expand (Conv2D)       (None, 8, 8, 960)    153600      ['block_13_project_BN[0][0]']    
                                                                                                  
 block_14_expand_BN (BatchNorma  (None, 8, 8, 960)   3840        ['block_14_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_14_expand_relu (ReLU)    (None, 8, 8, 960)    0           ['block_14_expand_BN[0][0]']     
                                                                                                  
 block_14_depthwise (DepthwiseC  (None, 8, 8, 960)   8640        ['block_14_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_14_depthwise_BN (BatchNo  (None, 8, 8, 960)   3840        ['block_14_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_14_depthwise_relu (ReLU)  (None, 8, 8, 960)   0           ['block_14_depthwise_BN[0][0]']  
                                                                                                  
 block_14_project (Conv2D)      (None, 8, 8, 160)    153600      ['block_14_depthwise_relu[0][0]']
                                                                                                  
 block_14_project_BN (BatchNorm  (None, 8, 8, 160)   640         ['block_14_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_14_add (Add)             (None, 8, 8, 160)    0           ['block_13_project_BN[0][0]',    
                                                                  'block_14_project_BN[0][0]']    
                                                                                                  
 block_15_expand (Conv2D)       (None, 8, 8, 960)    153600      ['block_14_add[0][0]']           
                                                                                                  
 block_15_expand_BN (BatchNorma  (None, 8, 8, 960)   3840        ['block_15_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_15_expand_relu (ReLU)    (None, 8, 8, 960)    0           ['block_15_expand_BN[0][0]']     
                                                                                                  
 block_15_depthwise (DepthwiseC  (None, 8, 8, 960)   8640        ['block_15_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_15_depthwise_BN (BatchNo  (None, 8, 8, 960)   3840        ['block_15_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_15_depthwise_relu (ReLU)  (None, 8, 8, 960)   0           ['block_15_depthwise_BN[0][0]']  
                                                                                                  
 block_15_project (Conv2D)      (None, 8, 8, 160)    153600      ['block_15_depthwise_relu[0][0]']
                                                                                                  
 block_15_project_BN (BatchNorm  (None, 8, 8, 160)   640         ['block_15_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_15_add (Add)             (None, 8, 8, 160)    0           ['block_14_add[0][0]',           
                                                                  'block_15_project_BN[0][0]']    
                                                                                                  
 block_16_expand (Conv2D)       (None, 8, 8, 960)    153600      ['block_15_add[0][0]']           
                                                                                                  
 block_16_expand_BN (BatchNorma  (None, 8, 8, 960)   3840        ['block_16_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_16_expand_relu (ReLU)    (None, 8, 8, 960)    0           ['block_16_expand_BN[0][0]']     
                                                                                                  
 block_16_depthwise (DepthwiseC  (None, 8, 8, 960)   8640        ['block_16_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_16_depthwise_BN (BatchNo  (None, 8, 8, 960)   3840        ['block_16_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_16_depthwise_relu (ReLU)  (None, 8, 8, 960)   0           ['block_16_depthwise_BN[0][0]']  
                                                                                                  
 block_16_project (Conv2D)      (None, 8, 8, 320)    307200      ['block_16_depthwise_relu[0][0]']
                                                                                                  
 block_16_project_BN (BatchNorm  (None, 8, 8, 320)   1280        ['block_16_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 Conv_1 (Conv2D)                (None, 8, 8, 1280)   409600      ['block_16_project_BN[0][0]']    
                                                                                                  
 Conv_1_bn (BatchNormalization)  (None, 8, 8, 1280)  5120        ['Conv_1[0][0]']                 
                                                                                                  
 out_relu (ReLU)                (None, 8, 8, 1280)   0           ['Conv_1_bn[0][0]']              
                                                                                                  
==================================================================================================
Total params: 2,257,984
Trainable params: 0
Non-trainable params: 2,257,984
__________________________________________________________________________________________________
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 mobilenetv2_1.00_224 (Funct  (None, 8, 8, 1280)       2257984   
 ional)                                                          
                                                                 
 global_average_pooling2d (G  (None, 1280)             0         
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 128)               163968    
                                                                 
 batch_normalization (BatchN  (None, 128)              512       
 ormalization)                                                   
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 3)                 387       
                                                                 
=================================================================
Total params: 2,422,851
Trainable params: 164,611
Non-trainable params: 2,258,240
_________________________________________________________________
2022-12-06 23:13:54.468624: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /device:GPU:0 with 9436 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3080 Ti, pci bus id: 0000:01:00.0, compute capability: 8.6
Found GPU at: /device:GPU:0
Epoch 1/150
2022-12-06 23:13:58.490778: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
2022-12-06 23:14:00.508206: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
95/95 [==============================] - 38s 341ms/step - loss: 0.3483 - precision: 0.7001 - recall: 0.9215 - accuracy: 0.8834 - true_negatives: 19457.0000 - true_positives: 11170.0000 - false_negatives: 951.0000 - false_positives: 4785.0000 - specificity: 0.7013 - f_measure: 0.7962 - f1_score: 0.7971 - val_loss: 0.1917 - val_precision: 0.8834 - val_recall: 0.9406 - val_accuracy: 0.9426 - val_true_negatives: 2842.0000 - val_true_positives: 1425.0000 - val_false_negatives: 90.0000 - val_false_positives: 188.0000 - val_specificity: 0.8281 - val_f_measure: 0.8806 - val_f1_score: 0.9110 - lr: 0.0010
Epoch 2/150
95/95 [==============================] - 28s 297ms/step - loss: 0.1650 - precision: 0.7788 - recall: 0.9756 - accuracy: 0.9426 - true_negatives: 20884.0000 - true_positives: 11825.0000 - false_negatives: 296.0000 - false_positives: 3358.0000 - specificity: 0.7638 - f_measure: 0.8567 - f1_score: 0.8663 - val_loss: 0.1593 - val_precision: 0.8266 - val_recall: 0.9756 - val_accuracy: 0.9459 - val_true_negatives: 2720.0000 - val_true_positives: 1478.0000 - val_false_negatives: 37.0000 - val_false_positives: 310.0000 - val_specificity: 0.7887 - val_f_measure: 0.8723 - val_f1_score: 0.8954 - lr: 0.0010
Epoch 3/150
95/95 [==============================] - 28s 293ms/step - loss: 0.1346 - precision: 0.8045 - recall: 0.9808 - accuracy: 0.9521 - true_negatives: 21353.0000 - true_positives: 11888.0000 - false_negatives: 233.0000 - false_positives: 2889.0000 - specificity: 0.7932 - f_measure: 0.8770 - f1_score: 0.8842 - val_loss: 0.1267 - val_precision: 0.8468 - val_recall: 0.9848 - val_accuracy: 0.9465 - val_true_negatives: 2760.0000 - val_true_positives: 1492.0000 - val_false_negatives: 23.0000 - val_false_positives: 270.0000 - val_specificity: 0.8452 - val_f_measure: 0.9097 - val_f1_score: 0.9111 - lr: 0.0010
Epoch 4/150
95/95 [==============================] - 28s 292ms/step - loss: 0.1139 - precision: 0.8117 - recall: 0.9874 - accuracy: 0.9591 - true_negatives: 21465.0000 - true_positives: 11968.0000 - false_negatives: 153.0000 - false_positives: 2777.0000 - specificity: 0.8144 - f_measure: 0.8925 - f1_score: 0.8911 - val_loss: 0.1266 - val_precision: 0.7915 - val_recall: 0.9875 - val_accuracy: 0.9525 - val_true_negatives: 2636.0000 - val_true_positives: 1496.0000 - val_false_negatives: 19.0000 - val_false_positives: 394.0000 - val_specificity: 0.7964 - val_f_measure: 0.8816 - val_f1_score: 0.8792 - lr: 0.0010
Epoch 5/150
95/95 [==============================] - 28s 293ms/step - loss: 0.1041 - precision: 0.8196 - recall: 0.9901 - accuracy: 0.9620 - true_negatives: 21600.0000 - true_positives: 12001.0000 - false_negatives: 120.0000 - false_positives: 2642.0000 - specificity: 0.8236 - f_measure: 0.8991 - f1_score: 0.8971 - val_loss: 0.1154 - val_precision: 0.8204 - val_recall: 0.9861 - val_accuracy: 0.9571 - val_true_negatives: 2703.0000 - val_true_positives: 1494.0000 - val_false_negatives: 21.0000 - val_false_positives: 327.0000 - val_specificity: 0.8303 - val_f_measure: 0.9015 - val_f1_score: 0.8962 - lr: 0.0010
Epoch 6/150
95/95 [==============================] - 27s 288ms/step - loss: 0.0899 - precision: 0.8316 - recall: 0.9936 - accuracy: 0.9672 - true_negatives: 21803.0000 - true_positives: 12044.0000 - false_negatives: 77.0000 - false_positives: 2439.0000 - specificity: 0.8368 - f_measure: 0.9084 - f1_score: 0.9057 - val_loss: 0.1433 - val_precision: 0.7755 - val_recall: 0.9828 - val_accuracy: 0.9459 - val_true_negatives: 2599.0000 - val_true_positives: 1489.0000 - val_false_negatives: 26.0000 - val_false_positives: 431.0000 - val_specificity: 0.7992 - val_f_measure: 0.8814 - val_f1_score: 0.8671 - lr: 0.0010
Epoch 7/150
95/95 [==============================] - 28s 296ms/step - loss: 0.0805 - precision: 0.8362 - recall: 0.9940 - accuracy: 0.9711 - true_negatives: 21882.0000 - true_positives: 12048.0000 - false_negatives: 73.0000 - false_positives: 2360.0000 - specificity: 0.8432 - f_measure: 0.9123 - f1_score: 0.9084 - val_loss: 0.1023 - val_precision: 0.8462 - val_recall: 0.9881 - val_accuracy: 0.9644 - val_true_negatives: 2758.0000 - val_true_positives: 1497.0000 - val_false_negatives: 18.0000 - val_false_positives: 272.0000 - val_specificity: 0.8384 - val_f_measure: 0.9071 - val_f1_score: 0.9116 - lr: 0.0010
Epoch 8/150
95/95 [==============================] - 27s 289ms/step - loss: 0.0751 - precision: 0.8477 - recall: 0.9939 - accuracy: 0.9733 - true_negatives: 22078.0000 - true_positives: 12047.0000 - false_negatives: 74.0000 - false_positives: 2164.0000 - specificity: 0.8521 - f_measure: 0.9175 - f1_score: 0.9152 - val_loss: 0.2087 - val_precision: 0.7210 - val_recall: 0.9875 - val_accuracy: 0.9228 - val_true_negatives: 2451.0000 - val_true_positives: 1496.0000 - val_false_negatives: 19.0000 - val_false_positives: 579.0000 - val_specificity: 0.7806 - val_f_measure: 0.8718 - val_f1_score: 0.8338 - lr: 0.0010
Epoch 9/150
95/95 [==============================] - ETA: 0s - loss: 0.0688 - precision: 0.8461 - recall: 0.9952 - accuracy: 0.9756 - true_negatives: 22047.0000 - true_positives: 12063.0000 - false_negatives: 58.0000 - false_positives: 2195.0000 - specificity: 0.8553 - f_measure: 0.9199 - f1_score: 0.9148
Epoch 9: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
95/95 [==============================] - 28s 292ms/step - loss: 0.0688 - precision: 0.8461 - recall: 0.9952 - accuracy: 0.9756 - true_negatives: 22047.0000 - true_positives: 12063.0000 - false_negatives: 58.0000 - false_positives: 2195.0000 - specificity: 0.8553 - f_measure: 0.9199 - f1_score: 0.9148 - val_loss: 0.1859 - val_precision: 0.8541 - val_recall: 0.9776 - val_accuracy: 0.9353 - val_true_negatives: 2777.0000 - val_true_positives: 1481.0000 - val_false_negatives: 34.0000 - val_false_positives: 253.0000 - val_specificity: 0.8774 - val_f_measure: 0.9247 - val_f1_score: 0.9114 - lr: 0.0010
Epoch 10/150
95/95 [==============================] - 27s 287ms/step - loss: 0.0551 - precision: 0.8589 - recall: 0.9966 - accuracy: 0.9803 - true_negatives: 22257.0000 - true_positives: 12080.0000 - false_negatives: 41.0000 - false_positives: 1985.0000 - specificity: 0.8641 - f_measure: 0.9256 - f1_score: 0.9229 - val_loss: 0.0967 - val_precision: 0.8690 - val_recall: 0.9848 - val_accuracy: 0.9637 - val_true_negatives: 2805.0000 - val_true_positives: 1492.0000 - val_false_negatives: 23.0000 - val_false_positives: 225.0000 - val_specificity: 0.8631 - val_f_measure: 0.9199 - val_f1_score: 0.9238 - lr: 5.0000e-04
Epoch 11/150
95/95 [==============================] - 27s 288ms/step - loss: 0.0465 - precision: 0.8694 - recall: 0.9975 - accuracy: 0.9842 - true_negatives: 22426.0000 - true_positives: 12091.0000 - false_negatives: 30.0000 - false_positives: 1816.0000 - specificity: 0.8713 - f_measure: 0.9301 - f1_score: 0.9292 - val_loss: 0.0940 - val_precision: 0.8492 - val_recall: 0.9921 - val_accuracy: 0.9670 - val_true_negatives: 2763.0000 - val_true_positives: 1503.0000 - val_false_negatives: 12.0000 - val_false_positives: 267.0000 - val_specificity: 0.8574 - val_f_measure: 0.9197 - val_f1_score: 0.9154 - lr: 5.0000e-04
Epoch 12/150
95/95 [==============================] - 28s 289ms/step - loss: 0.0470 - precision: 0.8634 - recall: 0.9969 - accuracy: 0.9832 - true_negatives: 22330.0000 - true_positives: 12083.0000 - false_negatives: 38.0000 - false_positives: 1912.0000 - specificity: 0.8699 - f_measure: 0.9291 - f1_score: 0.9255 - val_loss: 0.1120 - val_precision: 0.8358 - val_recall: 0.9881 - val_accuracy: 0.9611 - val_true_negatives: 2736.0000 - val_true_positives: 1497.0000 - val_false_negatives: 18.0000 - val_false_positives: 294.0000 - val_specificity: 0.8434 - val_f_measure: 0.9101 - val_f1_score: 0.9062 - lr: 5.0000e-04
Epoch 13/150
95/95 [==============================] - ETA: 0s - loss: 0.0367 - precision: 0.8667 - recall: 0.9989 - accuracy: 0.9884 - true_negatives: 22379.0000 - true_positives: 12108.0000 - false_negatives: 13.0000 - false_positives: 1863.0000 - specificity: 0.8743 - f_measure: 0.9324 - f1_score: 0.9283
Epoch 13: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
95/95 [==============================] - 27s 285ms/step - loss: 0.0367 - precision: 0.8667 - recall: 0.9989 - accuracy: 0.9884 - true_negatives: 22379.0000 - true_positives: 12108.0000 - false_negatives: 13.0000 - false_positives: 1863.0000 - specificity: 0.8743 - f_measure: 0.9324 - f1_score: 0.9283 - val_loss: 0.0947 - val_precision: 0.8228 - val_recall: 0.9927 - val_accuracy: 0.9663 - val_true_negatives: 2706.0000 - val_true_positives: 1504.0000 - val_false_negatives: 11.0000 - val_false_positives: 324.0000 - val_specificity: 0.8508 - val_f_measure: 0.9163 - val_f1_score: 0.8998 - lr: 5.0000e-04
Epoch 14/150
95/95 [==============================] - 28s 293ms/step - loss: 0.0332 - precision: 0.8728 - recall: 0.9988 - accuracy: 0.9901 - true_negatives: 22477.0000 - true_positives: 12106.0000 - false_negatives: 15.0000 - false_positives: 1765.0000 - specificity: 0.8781 - f_measure: 0.9345 - f1_score: 0.9317 - val_loss: 0.0937 - val_precision: 0.8619 - val_recall: 0.9888 - val_accuracy: 0.9630 - val_true_negatives: 2790.0000 - val_true_positives: 1498.0000 - val_false_negatives: 17.0000 - val_false_positives: 240.0000 - val_specificity: 0.8714 - val_f_measure: 0.9264 - val_f1_score: 0.9213 - lr: 2.5000e-04
Epoch 15/150
95/95 [==============================] - 28s 291ms/step - loss: 0.0289 - precision: 0.8764 - recall: 0.9990 - accuracy: 0.9917 - true_negatives: 22534.0000 - true_positives: 12109.0000 - false_negatives: 12.0000 - false_positives: 1708.0000 - specificity: 0.8809 - f_measure: 0.9362 - f1_score: 0.9340 - val_loss: 0.0948 - val_precision: 0.8681 - val_recall: 0.9908 - val_accuracy: 0.9617 - val_true_negatives: 2802.0000 - val_true_positives: 1501.0000 - val_false_negatives: 14.0000 - val_false_positives: 228.0000 - val_specificity: 0.8731 - val_f_measure: 0.9282 - val_f1_score: 0.9255 - lr: 2.5000e-04
Epoch 15: early stopping
time: [38.04026412963867, 28.22040343284607, 27.89033603668213, 27.75237250328064, 27.849269151687622, 27.381730794906616, 28.16930603981018, 27.389326333999634, 27.789839506149292, 27.33990240097046, 27.440694332122803, 27.50923800468445, 27.14223837852478, 27.883106231689453, 27.715942859649658]
