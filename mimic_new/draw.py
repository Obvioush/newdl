import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 2.6),dpi=500)
plt.figure(1)
ax1 = plt.subplot(121)
model = ('NKAM', 'KAME', 'GRAM', 'Dipole', 'RNN+', 'RNN')
Recall_top5 = [0.328,0.2931,0.2911,0.2773,0.2651,0.2635]
model_color=['tab:red','tab:green','tab:blue', 'tab:purple', 'tab:orange', 'tab:grey']
plt.ylim(0.24,0.34)
plt.ylabel('Recall@5')
plt.bar(model, Recall_top5,color=model_color)
plt.title('MIMIC-III数据集')

ax2 = plt.subplot(122)
# plt.figure(figsize=(4, 2.6),dpi=300)
model = ('NKAM', 'KAME', 'GRAM', 'Dipole', 'RNN+', 'RNN')
Precision_top5 = [0.6976,0.6314,0.6256,0.6002,0.5777,0.5831]
model_color=['tab:red','tab:green','tab:blue', 'tab:purple', 'tab:orange', 'tab:grey']
plt.ylim(0.50,0.75)
plt.ylabel('Precision@5')
plt.bar(model, Precision_top5,color=model_color)
plt.title('MIMIC-III数据集')

plt.show()