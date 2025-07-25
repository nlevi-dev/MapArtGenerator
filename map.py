import os
import matplotlib.pyplot as plt
import osmnx as ox
import geopandas as gpd
import numpy as np

HIGHWAY_RANK = {
    "motorway": 1,
    "motorway_link": 2,
    "trunk": 2,
    "trunk_link": 3,
    "primary": 3,
    "primary_link": 4,
    "secondary": 4,
    "secondary_link": 5,
    "tertiary": 5,
    "tertiary_link": 6,
    "unclassified": 6,
    "road": 6,
    "residential": 7,
    "living_street": 8,
    "service": 9,
    "track": 10,
    "path": 11,
    "footway": 12,
    "cycleway": 12,
    "pedestrian": 12,
    "steps": 12,
    "services": 13,
    "corridor": 13,
    "bus_stop": 13,
    "bridleway": 13,
    "elevator": 13,
    "rest_area": 13,
}

def min_rank(tags):
    if not isinstance(tags, (list, tuple, set)):
        tags = [tags]
    return min(HIGHWAY_RANK.get(t, 14) for t in tags)

bbox = (18.889618, 47.321138, 19.243927, 47.636709)
filename = str(bbox)+".gpkg"

if not os.path.isfile(filename):
    graph = ox.graph_from_bbox(bbox, network_type="all", simplify=True, retain_all=True, truncate_by_edge=False)
    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    edges["rank"] = edges["highway"].apply(min_rank)
    edges = edges[["geometry", "rank"]]
    edges.to_file(filename, layer='edges', driver="GPKG")
else:
    edges = gpd.read_file(filename, layer='edges')

print(edges["geometry"][0].coords[1])

# ig, ax = plt.subplots(figsize=(8, 8))
# ax.set_aspect("equal")
# ax.axis("off")

# edges.plot(ax=ax, linewidth=0.6, color="black")

# plt.show()