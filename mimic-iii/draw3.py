import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


if __name__ == '__main__':
    data = {}
    data['Dipole'] = [0.2792,0.2813,0.2791,0.2813,0.2820]
    data['GRAM'] = [0.2863,0.2928,0.2930,0.2942,0.2921]
    data['KAME'] = [0.2945,0.2975,0.2970,0.2966,0.2943]
    data['NKAM'] = [0.3229,0.3282,0.3276,0.3002,0.2963]

    plt.rcParams["figure.dpi"] = 140
    plt.style.use('ggplot')

    x_list = [64,128,192,256,320]
    plt.plot(x_list,data['Dipole'], marker='h', label='Dipole', markersize=8, color='tab:purple')
    plt.plot(x_list, data['GRAM'], marker='s', label='GRAM', markersize=8, color='tab:red')
    plt.plot(x_list, data['KAME'], marker='*', label='KAME', markersize=8, color='tab:blue')
    plt.plot(x_list, data['NKAM'], marker='^', label='NKAM', markersize=8, color='tab:grey')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.grid(True)
    # plt.axis([64, 320, min_value, max_value])
    x_major_locator = MultipleLocator(64)
    ax = plt.subplot(111)
    ax.xaxis.set_major_locator(x_major_locator)

    plt.xlim(56,328)
    plt.ylabel('Recall@5')
    plt.xlabel('GRU中不同维度的隐状态大小')
    plt.title('MIMIC-III数据集')
    plt.legend()
    plt.show()

