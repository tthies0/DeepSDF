import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
classes = ['Bench', 'Boat', 'Car', 'Chair', 'Lamp', 'Aircraft', 'Rifle', 'Sofa', 'Table']

errors_model1 = [7.549098032640969e-05, 0.0005683137644854065, 0.0003708119937060243, 
                 5.61945845323645e-05, 0.0006774831874314547, 0.00022247809626451103, 
                 0.0008116891008582599, 0.0019711441540932895, 0.00011472144327236269]
errors_model2 = [5.025041187238788e-05, 0.0005151195620283486, 0.0001599815290486968, 
                 3.659701798156974e-05, 0.00036450996531698596, 5.1966955582795316e-05, 
                 0.00010577413022401715, 3.574322590603016e-05, 8.817288547297998e-05]

# Create a tidy DataFrame for seaborn
df = pd.DataFrame({
    "Class": classes * 2,
    "Chamfer Distance": errors_model1 + errors_model2,
    "Model": ["DeepSDF"] * len(classes) + ["Ours (Class Embeddings)"] * len(classes)
})

# Set seaborn theme and palette
sns.set_theme(style="whitegrid")
palette = "Set2"

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df, x="Class", y="Chamfer Distance", hue="Model", palette=palette)

# Formatting: use scientific notation and set y-axis tick parameters
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# Set the offset text (exponent) on the y-axis to bold and increase fontsize
offset_text = ax.yaxis.get_offset_text()
offset_text.set_fontweight("bold")
offset_text.set_fontsize(14)

plt.ylabel('Chamfer Distance to GT', fontsize=18, fontweight='bold')
plt.xlabel('')  # Remove label on the x-axis
plt.xticks(rotation=45, ha='right', fontsize=18, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")
plt.legend(fontsize=18)

plt.tight_layout()
plt.savefig("chamfer_error.pdf", format='pdf')
plt.show()
