import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.decomposition import PCA

from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

G=nx.Graph()
vertices = list(range(1, 16))

# Create an array of edges
edges = []


rng1=random.Random(20)
for _ in range(20):
    # Pick two different random vertices

    v1, v2 = rng1.sample(vertices, 2)
    # Add the edge if it doesn't already exist
    if (v1, v2) not in edges and (v2, v1) not in edges:
        edges.append((v1, v2))

# Create an empty graph
G = nx.Graph()

# Add nodes from the vertices array
G.add_nodes_from(vertices)

# Add edges from the edges array
G.add_edges_from(edges)

# Draw the graph
print("graph 1")
nx.draw(G, with_labels=True)

# Display the graph
plt.show()

def get_randomwalk(node, path_length):
    random_walk = [node]

    for i in range(path_length - 1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node

    return random_walk

# get list of all nodes from the graph
all_nodes = list(G.nodes())

random_walks = []
for n in tqdm(all_nodes):
    for i in range(50):
        random_walks.append(get_randomwalk(n, 3))




# train skip-gram (word2vec) model
model = Word2Vec(window = 4, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(random_walks, progress_per=2)

model.train(random_walks, total_examples = model.corpus_count, epochs=100, report_delay=1)

print(model.wv.most_similar(positive=[1]))

def plot_nodes(word_list):
    X = model.wv[word_list]

    # reduce dimensions to 2
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    plt.figure(figsize=(12, 9))
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.show()


from sklearn.manifold import TSNE

def visualize_embeddings(model):
    # Get the embeddings from the model
    embeddings = model.wv.vectors

    # Use t-SNE to reduce the dimensionality of the embeddings to 2
    tsne = TSNE(n_components=2, perplexity=5,random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Create a scatter plot of the reduced embeddings
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    # Annotate the points with their corresponding words
    for i, word in enumerate(model.wv.index_to_key):
        plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    plt.show()


terms = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#plot_nodes(terms)
#visualize_embeddings(model)

embeddings = model.wv.vectors
print(embeddings)

from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)

G_reconstructed = nx.Graph()
# Add nodes
for i in range(embeddings.shape[0]):
    G_reconstructed.add_node(i)

# Add edges based on similarity
for i in range(similarity_matrix.shape[0]):
    for j in range(i+1, similarity_matrix.shape[1]):
        # You can add a threshold to control which edges are added
        if similarity_matrix[i, j] > 0.8:
            G_reconstructed.add_edge(i, j)


# Now G_reconstructed is a graph reconstructed from the embeddings
nx.draw(G_reconstructed, with_labels=True)
plt.show()