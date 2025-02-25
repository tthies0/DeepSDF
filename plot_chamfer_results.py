import matplotlib.pyplot as plt
import numpy as np

classes = ['Bench', 'Boat', 'Car', 'Chair', 'Lamp', 'Aircraft', 'Rifle', 'Sofa', 'Table']

errors_model1 = [7.549098032640969e-05, 0.0005683137644854065, 0.0003708119937060243, 5.61945845323645e-05, 0.0006774831874314547, 0.00022247809626451103, 0.0008116891008582599, 0.0019711441540932895, 0.00011472144327236269]
errors_model2 = [5.025041187238788e-05, 0.0005151195620283486, 0.0001599815290486968, 3.659701798156974e-05, 0.00036450996531698596, 5.1966955582795316e-05, 0.00010577413022401715, 3.574322590603016e-05, 8.817288547297998e-05]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, errors_model1, width, label='DeepSDF')
bars2 = ax.bar(x + width/2, errors_model2, width, label='Ours (Class Embeddings)')

ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
offset_text = ax.yaxis.get_offset_text()
offset_text.set_fontsize(16)
offset_text.set_fontweight('bold')
ax.set_ylabel('Chamfer Distance to GT', fontsize=18, fontweight='bold')
#ax.set_title('Error by Class for Two Machine Learning Models', fontsize=20, fontweight='bold')

ax.set_xticks(x)
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
    tick.set_fontsize(14)
    
ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=18, fontweight="bold")
ax.legend(fontsize=18)

"""
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height*1e4:.1f}e-4',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),  # Offset text by 3 points vertically
                textcoords="offset points",
                ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height*1e4:.1f}e-4',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')
"""
plt.tight_layout()
plt.savefig("chamfer_error.pdf", format='pdf')
plt.show()
