import math
import os

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import scipy.misc as mi
# import skimage.measure
from PIL import Image

# '''
# 函数：拼接多张图为一张图
#     param1: int 需要拼接出来的图像大小
#     param2: int
# '''
# def image_montage(weight,height,path,save_path):
#     path_list = os.listdir(path)
#     sum_image = np.zeros((weight,height))
#     X = 0
#     Y = 0
#     tmp = 0
#     for files_root in path_list:
#         files = os.listdir(path+'/'+files_root)
#         for file in files:
#             real_image = mi.imread(path+'/'+files_root+'/'+file)
#             [rows,cols] = real_image.shape
#             x_mins = 0
#             y_mins = 0
#             if((X+rows)>weight):
#                 x_mins = (X+rows) - weight
#             if((Y+rows)>height):
#                 y_mins = (Y+rows) - height
#             for i in range(x_mins,rows):
#                 for j in range(y_mins,cols):
#                     sum_image[X+i-x_mins][Y+j-y_mins] = real_image[i][j]
#             Y += cols
#             tmp = rows
#         Y = 0
#         X += tmp
#     mi.imsave(save_path,sum_image)
#     print('Finish!!!')
#
# """
# 方法：保存mrc格式的文件为常见图片格式
#     param1: String 源mrc文件的读取地址
#     param2: String 常见图片格式的目的地址
#     param3: String 源mrc文件格式(mrc or mrcs)
#     param4: String 输入转化成的图片格式
#
# """
# def save_mrcfile_to_image(read_path,save_path,image_type,save_type):
#     paths=[]
#     paths = os.listdir(read_path)
#     if(image_type=='mrc'):
#         for i in paths:
#             print(i)
#             with mrcfile.open(read_path+i) as mrc:
#                 tmp = mrc.data
#                 print(tmp)
#                 tmp = (tmp - mrc.data.min())/(mrc.data.max()-mrc.data.min())
#                 # tmp = tf.image.convert_image_dtype(
#                 #     tmp,
#                 #     dtype=tf.uint8,
#                 # )
#                 print(tmp)
#                 mi.imsave(save_path+i[:-4]+'.'+save_type, tmp)
#     if(image_type=='mrcs'):
#         for i in paths:
#             with mrcfile.open(read_path+i) as mrcs:
#                 index = 0
#                 tmp = mrcs.data
#                 for mrc in tmp:
#                     tmp = mrc
#                     tmp = (tmp - mrc.min())/(mrc.max()-mrc.min())
#                     mi.imsave(save_path+i+str(index)+'.'+save_type, tmp)
#                     index += 1
#     print('save_mrcfile_to_image finish!!!')
#
# """
# 方法：绘制直方统计图(只支持15个数据)
#     param1: list 统计图的X轴显示值(n*1)
#     param2: list 统计图的Y轴显示值(n*1)
#     param3: float 直方图的图像宽度(0~1,推荐0.7)
#     param4: String 直方图X轴的解释
#     param5: String 直方图Y轴的解释
#     param6: String 直方图的标题
#
# """
# def draw_bar_chart(X_label,X_data,bar_weight,X_name,Y_name,titles,size,save):
#     colors = ['#666600','#669966','#CC9966','#999999','#ffa631',
#               '#a3d900','#3d3b4f','#896c39','#758a99','#bbcdc5',
#               '#eedeb0','#d6ecf0','#99CC33','#696969','#DA70D6']
#     fig = plt.figure(figsize=size) #初始化一个画布
#     plt.bar(x=range(len(X_label)),height=X_data,width=bar_weight,color=colors)
#     plt.xticks(range(len(X_label)),X_label)
#     plt.xlabel(X_name)
#     plt.ylabel(Y_name)
#     plt.title(titles)
#     if(save):
#         fig.savefig(titles+".jpg")
#     plt.show()
#     plt.close()
#
# """
# 方法：绘画统计图(只支持15个数据)
#     param1: list 统计图的X轴显示值(n*1)
#     param2: list 统计图的Y轴显示值(传入时须封装成list集合)
#     param3: float 直方图的图像宽度
#     param4: String 直方图X轴的解释
#     param5: String 直方图Y轴的解释
#     param6: String 直方图的标题
# """
# def draw_statistical_chart(X_label,X_data,X_name,Y_name,titles,size,save):
#     fig = plt.figure(figsize = size)
#     colors = ['#666600','#669966','#CC9966','#999999','#ffa631',
#               '#a3d900','#3d3b4f','#896c39','#758a99','#bbcdc5',
#               '#eedeb0','#d6ecf0','#99CC33','#696969','#DA70D6']
#     for i in range(len(X_data)):
#         plt.plot(X_data[i], color = colors[i], label=X_label[i],linewidth=0.8)
#     plt.legend(loc='upper right')
#     plt.xlabel(X_name)
#     plt.ylabel(Y_name)
#     plt.title(titles)
#     if(save):
#         fig.savefig(titles+".jpg")
#     plt.show()
#     plt.close()
#
# """
# 方法: 裁剪图片
#     param1: numpy 数组图片
#     param2: int   裁剪图片的宽
#     param3: int   裁剪图片的高
#     return: numpy 裁剪后的数组图片
# """
def image_crop_function(image,x_size,y_size):
    [x,y] = image.shape
    # r_x = np.random.randint(0,x-x_size)
    # r_y = np.random.randint(0,y-y_size)

    # if(x_size>image.shape[0]):
    #     r_x = image.shape[0]
    # if(y_size>image.shape[1]):
    #     r_y = image.shape[1]
    image_crop = image[0:x_size,1855:y_size]
    return image_crop

"""
方法： 调整图片大小
    param1: numpy 数组图片
    param2: int   图片的宽调整为x_size
    param3: int   图片的高调整为y_size
    return: numpy 调整后的数组图片
"""
def image_resize_fuction(image,x_size,y_size):
    images = Image.fromarray(np.float32(image))
    images_tmp = images.resize((x_size,y_size),Image.ANTIALIAS)
    images_array = np.asarray(images_tmp)
    return images_array


# """
# 方法： 添加噪声
#     param1: numpy  数组图片(0~1)
#     param2: String 噪声类型(Gaussian,Gaussians,Poisson)
#     param3: int    调整参数（Gaussian）
#     return: numpy  加噪后的数组图片
# """
# def image_addnoise_fuction(image,tpye_noise,var):
#     if(tpye_noise=='Gaussian'):
#         var = 0.001*var
#         image = np.array(image, dtype=float)
#         noise = np.random.normal(0, var ** 0.5, image.shape)
#         out = image + noise
#         if out.min() < 0:
#             low_clip = -1.
#         else:
#             low_clip = 0.
#         out = np.clip(out, low_clip, 1.0)
#         return out
#     if(tpye_noise=='Gaussians'):
#         GuassNoise = np.random.normal(0, var, image.shape)
#         image = image*255
#         noisyImg = image + GuassNoise # float type noisy image
#
#     #    cv2.normalize(noisyImg, noisyImg, 0, 255, cv2.NORM_MINMAX, dtype=-1)
#     #
#     #    noisyImg = noisyImg.astype(np.uint8)
#     #
#     #    cv2.imwrite('noisydog.png', noisyImg)
#     #
#     #    if cv2.imwrite('noisydog.png', noisyImg) == True:
#     #
#     #        print('Noise has been added to the original image.\n')
#     #
#     #        return noisyImg
#     #
#     #    else:
#     #
#     #        print('Error: adding noise failed.\n')
#     #
#     #        exit()
#
#         return noisyImg/255
#     if(tpye_noise=='Poisson'):
#         x = np.random.poisson(lam=300, size=(256,256))
#         noise = image+x
#
# """
# 方法： 计算两张图片的峰值信噪比（peak signal to noise rate,PSNR）
#     param1: numpy 数组图片(0~255)
#     param2: numpy 数组图片(0~255)
#     return: float 两张图片的PSNR
# """
# # img1=Image.open("./example/1_10089.tif")
# # img2=Image.open("./example/adjust/adjusted_10089.tif")
# # arr1 = np.array(img1)
# # arr2 = np.array(img2)
# def psnr(img1, img2):
#
#     mse = np.mean( (img1/255.0 - img2/255.0) ** 2 )
#     if mse < 1.0e-10:
#         return 100
#     PIXEL_MAX = 1
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#
# # print(psnr(arr1,arr2))
#
#
# """
# 方法： 计算两张图片的平均误差（Mean Square Error loss,MSE）
#     param1: numpy 数组图片(0~255)
#     param2: numpy 数组图片(0~255)
#     return: float 两张图片的MSE值
# """
# def mse(y_true,y_pred):
#     loss = (y_true-y_pred)**2
#     MSE_number = np.mean(loss)
#     return MSE_number
#
# """
# 方法： 计算两张图片的均方误差（Root Square Error loss,PMSE）
#     param1: numpy 数组图片(0~255)
#     param2: numpy 数组图片(0~255)
#     return: float 两张图片的MSE值
# """
# def pmse(y_true,y_pred):
#     loss = (y_true-y_pred)**2
#     MSE_number = (np.mean(loss))**0.5
#     return MSE_number
#
# """
# 方法： 计算两张图片的平均绝对误差（Mean Absolute Error,MAE）
#     param1: numpy 数组图片(0~255)
#     param2: numpy 数组图片(0~255)
#     return: float 两张图片的MAE值
# """
# def mae(y_true,y_pred):
#     loss = ((y_true-y_pred)**2)**0.5
#     MSE_number = np.mean(loss)
#     return MSE_number
#
# """
# 方法： 计算两张图片的结构相似性（Structural Similarity Index,SSIM）
#     param1: numpy 数组图片(0~255)
#     param2: numpy 数组图片(0~255)
#     return: float 两张图片的SSIM值
#     注：(详细使用方法)https://cloud.tencent.com/developer/section/1414961中的compare_ssim方法
# """
# def ssim(x,y):
#     return skimage.measure.compare_ssim(x, y, data_range=255, gaussian_weights=True)
#
# # image_crop_function(image,x_size,y_size)
# # save_mrcfile_to_image("img/","demon/","mrc","png")