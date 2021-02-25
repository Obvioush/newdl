import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
    # plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    mimic3 = np.array([0.3579, 0.3648, 0.3632, 0.3623, 0.3641, 0.3568, 0.3560, 0.3556])
    mimic4 = np.array([0.5112, 0.5123, 0.5124, 0.5133, 0.5126, 0.5121, 0.5111, 0.5115])

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1

    plt.figure(figsize=(8, 5), dpi=600)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    # plt.plot(x, mimic3, color="tab:red", marker='h', label="MIMIC-III Dataset", linewidth=1.5)
    plt.plot(x, mimic4, color="tab:blue", marker='o', label="MIMIC-IV Dataset", linewidth=1.5)



    group_labels = ['1', '2', '3', '4', '5', '6', '7', '8']  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=12)  # 默认字体大小为10
    plt.yticks(fontsize=12)
    plt.xlabel("The number of heads", fontsize=13)
    plt.ylabel("Code-level Accuracy@5", fontsize=13)
    plt.xlim(0.75, 8.25)  # 设置x轴的范围
    # plt.ylim(0.352, 0.368)
    plt.ylim(0.5105, 0.5140)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12)  # 设置图例字体的大小和粗细

    # plt.savefig('./m4_head.eps', dpi=600)
    plt.show()
