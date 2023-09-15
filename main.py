import os
import time
from yolox import yolox
import cv2
import torch
# 返回的结果为格式为result，srcimg为输出图片
onnx_path = '/root/zhangpeng-chengdu/yolox/exp/coco_CSPDarknet-s_640x640_renlian_small/model_best_renlian.onnx'  # onnx 模型的路径
img_path = './database/input_img'    # 图片的路径
sav_path = './database/save_img/'    # 图片保存的路径

conf_thr = 0.2  #表示置信度以下不予计算
input_size = (960,960)   # 图片推理尺寸 设置需要长和宽一致
img_show = True # 是否图片可视化,如果可视化后，图片保存在sav_path 路径
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')  # 判断是否使用GPU  cpu 和 GPU都可以调用  可以强行调用yolox 文件中providers参数

net = yolox(model=onnx_path,device = device,input_size = input_size,confThreshold=conf_thr)

# result 的字典形式为 yolox文件中的153行  其中bbox 表示目标框 分别表示框的左上角和右下角坐标
# score 表示置信度  category_id 表示检测的类别 其中类别的名称为 coco.names中

for filename in os.listdir(img_path):
    t0 = time.time()
    img_name = img_path +'/'+ filename
    srcimg = cv2.imread(img_name)
    if img_show:
        save_path = sav_path + filename
        srcimg, result = net.detect(srcimg, show=img_show)
        cv2.imwrite(save_path, srcimg)
    else:
        result = net.detect(srcimg, show=img_show)
    t1 = time.time()
    print('the gpu is cost time is', t1 - t0)







