import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load data from CSV files
data1 = pd.read_csv("./examples/sofas_lecture/Evaluation/460/chamfer.csv", skipinitialspace=True)
data2 = pd.read_csv("./examples/transformer_sofas/sofas_lecture/Evaluation/490/chamfer.csv", skipinitialspace=True)
data3 = pd.read_csv("./examples/transformer_sofas/sofas_lecture/Evaluation/2500/chamfer.csv", skipinitialspace=True)
data4 = pd.read_csv("./examples/one_hot_large_architecture_ipol/Evaluation/450/chamfer.csv", skipinitialspace=True)

# Extract chamfer distance values
model1 = data1["chamfer_dist"]
model2 = data2["chamfer_dist"]
model3 = data3["chamfer_dist"]
model4 = data4["chamfer_dist"]

# Combine data for seaborn
models = ["DeepSDF (460)"] * len(model1) + ["Transformer (460)"] * len(model2) + ["Transformer (2500)"] * len(model3) + ["Class Embeddings"] * len(model4)
values = list(model1) + list(model2) + list(model3) + list(model4)
df = pd.DataFrame({"Model": models, "Chamfer Distance": values})

# Set seaborn style
sns.set_theme(style="whitegrid")

# Boxplot visualization
plt.figure(figsize=(10, 4.5))
sns.boxplot(x="Model", y="Chamfer Distance", data=df, palette="Set2")

plt.yscale("log")  # Set logarithmic scale on y-axis
plt.ylabel("Chamfer Distance to GT (Logarithmic)", fontsize=14, fontweight="bold")
plt.xticks(fontsize=13, fontweight="bold")
plt.yticks(fontsize=13, fontweight="bold")
#plt.title("Chamfer Distance Across 135 Shape from Class \"Sofa\"", fontsize=16, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.6, which='both')  # Finer grid on y-axis
plt.xlabel("")  # Remove x-axis label

plt.tight_layout()
plt.savefig("chamfer_comparison.pdf", format="pdf")
plt.show()
