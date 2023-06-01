
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
import numpy as np


DATADIR = os.path.expanduser("~/brainhack/brainhack_project/dynamical_connectivity/generated_connectivity_data/")


# Choose a frequency band and a time window
freq_band = 'alpha'
time_window = 0  # adjust this to choose a different time window
method = 'wpli'
n_ROI = 360

# Get the connectivity matrix for this frequency band and time window
con_mat = np.load(DATADIR+f'{freq_band}_connectivity_matrices_{method}.npy')
print(con_mat.shape)



#CONNECTIVITY HEATMAP#

plt.figure(figsize=(10, 8))
sns.heatmap(con_mat, cmap='viridis')
plt.title(f'Connectivity Heatmap for {freq_band.capitalize()} Band, Time Window {time_window}')
plt.show()


#CONNECTIVITY NETWORK DIAGRAM#

con_mat = con_mat[time_window]

# Create a new graph
G = nx.Graph()

# Add nodes
for i in range(n_ROI):
    G.add_node(i)

# Add edges
for i in range(n_ROI):
    for j in range(i+1, n_ROI):  # only consider the upper triangle of the matrix to avoid double-counting edges
        # Add an edge between ROIs i and j, with weight equal to the connectivity between them
        G.add_edge(i, j, weight=con_mat[i, j])

# Draw the graph
pos = nx.circular_layout(G)  # This arranges nodes in a circle, you might want to replace this with spatial coordinates of your ROIs if you have them.
weights = nx.get_edge_attributes(G, 'weight')

nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color=weights.values(), width=2.0, edge_cmap=plt.cm.Blues)
plt.title(f'Connectivity Network for {freq_band.capitalize()} Band, Time Window {time_window}')
plt.show()
