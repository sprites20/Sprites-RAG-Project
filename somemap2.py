import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from config import location_point, origin, destination

"""
(lat, lon) format
"""
location_point = <location_point>
origin = <origin>
destination = <destination>

# Define the bounding box or the center and distance to get the street network
G = ox.graph_from_point(location_point, dist=16000, network_type='drive')

# Find the nearest nodes to the origin and destination points
origin_node = ox.distance.nearest_nodes(G, origin[1], origin[0])
destination_node = ox.distance.nearest_nodes(G, destination[1], destination[0])

# Find the shortest path between the nodes
route = nx.shortest_path(G, origin_node, destination_node, weight='length')
print("Found path")

print("plotting")
# Plot the route on the map
fig, ax = ox.plot_graph_route(G, route, node_size=0)

# Show the plot
plt.show()
