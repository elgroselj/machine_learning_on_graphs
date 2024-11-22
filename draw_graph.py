import networkx as nx
import matplotlib.pyplot as plt

# Define the edges based on the provided connections
# edges = [
#     ('races', 'circuits', 820),
#     ('circuits', 'races', 820),
#     ('constructor_standings', 'races', 10170),
#     ('races', 'constructor_standings', 10170),
#     ('constructor_standings', 'constructors', 10170),
#     ('constructors', 'constructor_standings', 10170),
#     ('standings', 'races', 28115),
#     ('races', 'standings', 28115),
#     ('standings', 'drivers', 28115),
#     ('drivers', 'standings', 28115),
#     ('constructor_results', 'races', 9408),
#     ('races', 'constructor_results', 9408),
#     ('constructor_results', 'constructors', 9408),
#     ('constructors', 'constructor_results', 9408),
#     ('results', 'races', 20323),
#     ('races', 'results', 20323),
#     ('results', 'drivers', 20323),
#     ('drivers', 'results', 20323),
#     ('results', 'constructors', 20323),
#     ('constructors', 'results', 20323),
#     ('qualifying', 'races', 4082),
#     ('races', 'qualifying', 4082),
#     ('qualifying', 'drivers', 4082),
#     ('drivers', 'qualifying', 4082),
#     ('qualifying', 'constructors', 4082),
#     ('constructors', 'qualifying', 4082)
# ]

edges = [
    ('races', 'circuits', 820),
    ('constructor_\nstandings', 'races', 10170),
    ('constructor_\nstandings', 'constructors', 10170),
    ('standings', 'races', 28115),
    ('standings', 'drivers', 28115),
    ('constructor_\nresults', 'races', 9408),
    ('constructor_\nresults', 'constructors', 9408),
    ('results', 'races', 20323),
    ('results', 'drivers', 20323),
    ('results', 'constructors', 20323),
    ('qualifying', 'races', 4082),
    ('qualifying', 'drivers', 4082),
    ('qualifying', 'constructors', 4082)
]

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])
    
G.add_edge("TRAIN_TABLE", "drivers")

# Draw the graph
pos = nx.circular_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.savefig("graph.png")