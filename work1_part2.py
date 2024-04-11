from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os.path
import jieba
from collections import Counter
import math

#文件夹路径
file_path = r'D:\Deeplearning'
#存放单独文件路径
filePaths=[]
#语料库序列
file_text = ""
# 读取停用词数据



with open('stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = file.read().splitlines()

for root, dirs, files in os.walk(file_path):
#遍历目录和其子目录，并返回一个三元组（root, dirs, files）的生成器
#root：目录路径。
#files：文件名列表。
    for name in files:
        if name.endswith(".txt"):  # 文件是否为txt文件
            filePath = os.path.join(root, name)#将文件名与路径相连
            filePaths.append(filePath)#存放所有文件的路径
            file1 = open(filePath, 'r', encoding='utf-8')#打开当前txt文件、只读
            text = file1.read()#读取当前文件的所有内容
            file1.close()#关闭文件
            file_text += text#将当前小说存入数据集中

file_text= file_text.replace("\n", "")
file_text = file_text.replace("\u3000", "")
file_text= file_text.replace(' ', "")
file_text = file_text.replace("本书来自www.cr173.com免费txt小说下载站", "")
file_text = file_text.replace("更多更新免费电子书请关注www.cr173.com", "")
words = jieba.cut(file_text)
# 过滤停顿词
filtered_words = [word for word in words if word not in stopwords]

#一元模型计算
def modle_1(text):
    #词频 定义一个字典
    frequence = {}
    num = 0
    for key in text:
        if key in frequence:
            frequence[key] += 1
        else:
            frequence[key] = 1
        num += 1

    entropy = 0
    for count in frequence.values():
        px = count / num
        entropy -= px * math.log(px, 2)
    return entropy


def modle_2(text):

    #二元词组
    xx_1 = [(text[i], text[i + 1]) for i in range(len(text) - 1)]
    xx_count = Counter(xx_1)
    num_xx = len(xx_1)

    #词频 定义一个字典
    frequence = {}
    num = 0
    entropy = 0
    for key in text:
        if key in frequence:
            frequence[key] += 1
        else:
            frequence[key] = 1
        num += 1
    #联合概率p(x,y)近似等与每个二元数组在语料库中出现的频率
    #条件概率p（x|y）近似等于每个二元词组在语料库中出现的频数与该二元词组的第一个词为词首的二元词组频数的比值
    for xx,count in xx_count.items():
        pxy = count / num_xx  #
        p_x_y = count / frequence[xx[0]]  # 条件概率p(x|y)
        entropy -= pxy * math.log(p_x_y, 2)
    return entropy

def modle_3(text):
    #联合概率P（x,y,z）近似等于每个三元词组在语料库中出现的频率
    #条件概率p（x|y,z）近似等于每个三元词组在语料库中出现的频数与该三元词组前两个词为词首的三元词组的频数的比值
    x_y = [(text[i], text[i+1]) for i in range(len(text)-1)]#二元词组
    bigram_count = Counter(x_y)
    xy_num = len(x_y)

    x_y_z = [(text[i], text[i + 1], text[i + 2]) for i in range(len(text) - 2)]  # 三元词组
    xyz_count = Counter(x_y_z)
    xyz_num = len(x_y_z)  # 三元组的总数

    entropy = 0
    for trigram, count in xyz_count.items():
        p_xyz = count / xyz_num  # p(x,y,z)
        p_xy_z = count / bigram_count[(trigram[0],trigram[1])]  # 计算条件概率
        entropy -= p_xyz * math.log(p_xy_z, 2)#条件概率p(x|y)
    return entropy



entropy1 = modle_1(text)
entropy2 = modle_2(text)
entropy3 = modle_3(text)
print("一元信息熵为:", entropy1)
print("二元信息熵为:", entropy2)
print("三元信息熵为:", entropy3)









