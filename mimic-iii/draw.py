import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 2.6),dpi=300)
plt.style.use('ggplot')
plt.figure(1)
ax1 = plt.subplot(121)
model = ('NKAM', 'KAME', 'GRAM', 'Dipole', 'RNN+', 'RNN')
Recall_top5 = [0.3282,0.2975,0.2928,0.2813,0.2733,0.2700]
model_color=['tab:red','tab:green','tab:blue', 'tab:purple', 'tab:orange', 'tab:grey']

# plt.grid(alpha=0.3)
plt.ylim(0.25,0.34)
plt.ylabel('Recall@5')
plt.bar(model, Recall_top5,color=model_color)
plt.title('MIMIC-III数据集')

ax2 = plt.subplot(122)
# plt.figure(figsize=(4, 2.6),dpi=300)
model = ('NKAM', 'KAME', 'GRAM', 'Dipole', 'RNN+', 'RNN')
Precision_top5 = [0.6984,0.6408,0.6270,0.6092,0.5950,0.5931]
model_color=['tab:red','tab:green','tab:blue', 'tab:purple', 'tab:orange', 'tab:grey']
# plt.style.use('ggplot')
# plt.grid(alpha=0.3)
plt.ylim(0.55,0.75)
plt.ylabel('Precision@5')
plt.bar(model, Precision_top5,color=model_color)
plt.title('MIMIC-III数据集')

plt.show()