import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer


# Open the text file
file = open("dataset/sample_sentences.txt", "r")

# Read all sentences
sentences = file.readlines()

# Close the file
file.close()


# Display sentences
for sentence in sentences:
    print(sentence.strip())


# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the sentences
embeddings = model.encode(sentences)

# Save the embeddings to a .npy file
np.save("dataset/embeddings.npy", embeddings)

print("Embeddings saved successfully.")


# Load the embeddings from the .npy file
loaded_embeddings = np.load("dataset/embeddings.npy")

print("Embedding shape:", loaded_embeddings.shape)


# Create Q, K, V matrices for attention mechanism
embedding_size = loaded_embeddings.shape[1]

W_Q = np.random.rand(embedding_size, embedding_size)
W_K = np.random.rand(embedding_size, embedding_size)
W_V = np.random.rand(embedding_size, embedding_size)


# Generate Q, K, V
Q = loaded_embeddings @ W_Q
K = loaded_embeddings @ W_K
V = loaded_embeddings @ W_V


print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)


# Calculate attention scores
scores = Q @ K.T


# Scale the attention scores
scaled_scores = scores / np.sqrt(embedding_size)

print("Scaled Attention Scores:")
print(scaled_scores)


# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Calculate attention weights
attention_weights = softmax(scaled_scores)


print("Attention Weights:")
print(attention_weights)

print(np.sum(attention_weights, axis=1))


# Calculate Final Attention Output
final_output = attention_weights @ V


print("Final Attention Output:")
print(final_output)


# Save the attention weights to a CSV file
weights_df = pd.DataFrame(attention_weights)

weights_df.to_csv(
    "dataset/attention_weights.csv",
    index=False
)

print("Attention weights saved successfully.")


# Save the final attention output to a CSV file
output_df = pd.DataFrame(final_output)

output_df.to_csv(
    "dataset/attention_output.csv",
    index=False
)

print("Attention output saved successfully.")


# Visualize Attention Using Heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    attention_weights,
    annot=True,
    fmt=".2f",
    cmap="Blues"
)

plt.title("Attention Weights Heatmap")

plt.xlabel("Key")
plt.ylabel("Query")

plt.tight_layout()

plt.savefig("dataset/attention_heatmap.png")

plt.show()

print("Attention heatmap saved successfully.")