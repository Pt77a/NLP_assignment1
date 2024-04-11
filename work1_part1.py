from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os.path
import jieba
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

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

# 统计词频
word_freq = Counter(filtered_words)

# 根据词频排序
word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

# 获取词频的排名和词频数
ranks = list(range(1, len(word_freq) + 1))
freqs = list(word_freq.values())

# 输出出现频率最高的前几个词语
num_top_words = 100
top_words = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:num_top_words])
print(top_words)


#创建画布
plt.figure(figsize=(8, 6))
plt.plot(np.log(ranks), np.log(freqs))
plt.xlabel('排名')
plt.ylabel('频率')
plt.title("Zipf's Law")
plt.show()

