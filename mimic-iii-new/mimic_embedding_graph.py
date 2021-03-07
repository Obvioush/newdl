import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
    # plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    x = np.array([50, 100, 150, 200, 250])
    mimic3 = np.array([0.316814, 0.327837, 0.328689, 0.328367, 0.310906])
    mimic4 = np.array([0.379753, 0.399345, 0.401661, 0.400115, 0.397090])

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1

    plt.figure(figsize=(9, 6), dpi=300)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.plot(x, mimic3, color='k', marker='h', label="MIMIC-III Dataset")
    plt.plot(x, mimic4, color='k', marker='^', label="MIMIC-IV Dataset")



    group_labels = ['50', '100', '150', '200', '250']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=20)  # 默认字体大小为10
    plt.yticks(fontsize=20)
    plt.xlabel("Embedding size", fontsize=20)
    plt.ylabel("Recall@5", fontsize=20)
    plt.xlim(45, 252)  # 设置x轴的范围
    plt.ylim(0.3, 0.41)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=20)  # 设置图例字体的大小和粗细


    plt.savefig('./mimic_new_embedding.eps', dpi=300)  # 建议保存为svg格式,再用在线转换工具转为矢量图emf后插入word中
    plt.show()
