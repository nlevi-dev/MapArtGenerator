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

rng = np.random.default_rng(36)
bbox = (18.889618, 47.301138, 19.243927, 47.636709)

filename = str(bbox)+".graphml"
if not os.path.isfile(filename):
    graph = ox.graph_from_bbox(bbox, network_type="all", simplify=True, retain_all=True, truncate_by_edge=False)
    ox.save_graphml(graph, filename)
else:
    graph = ox.load_graphml(filename)

edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
edges["rank"] = edges["highway"].apply(min_rank)
edges = edges[["geometry", "rank"]]
cx = np.array(edges["geometry"].centroid.x)
cy = np.array(edges["geometry"].centroid.y)
edges["cx"] = cx
edges["cy"] = cy
ds = np.min(np.array([np.abs(cx - bbox[0]), np.abs(cx - bbox[2]), np.abs(cy - bbox[1]), np.abs(cy - bbox[3])]), axis=0)
ds = ds / np.max(ds) * -1 + 1
edges["dist"] = ds


def grid(cx, cy, GRID=10):
    x_low, y_low, x_high, y_high = bbox
    x_edges = np.linspace(x_low, x_high, GRID + 1)
    y_edges = np.linspace(y_low, y_high, GRID + 1)
    h, _, _ = np.histogram2d(cy, cx, bins=[y_edges, x_edges])
    h = np.array(h, dtype=np.float64)
    return h, x_edges, y_edges

def grid_find(x, y, grid_count, x_edges, y_edges):
    ix = np.searchsorted(x_edges, x, side='right') - 1
    iy = np.searchsorted(y_edges, y, side='right') - 1
    ix = min(ix, grid_count.shape[1] - 1)
    iy = min(iy, grid_count.shape[0] - 1)
    return grid_count[iy, ix]

def filter_dropout(df, function):
    drop = []
    for i in range(len(df)):
        chance = function(df.iloc[i])
        drop.append(chance <= rng.random())
    drop = np.array(drop)
    return df[drop]

# normalize density
MIN_RANK = 7
GRID = 30
TARGET = 1000000
area = ((bbox[2] - bbox[0]) / GRID) * ((bbox[3] - bbox[1]) / GRID)
target = TARGET * area
d_edges = edges[edges["rank"] >= MIN_RANK]
d_grid, d_bound_x, d_bound_y = grid(np.array(d_edges["cx"]), np.array(d_edges["cy"]), GRID)
def d_dropout(row):
    rank = row["rank"]
    x = row["cx"]
    y = row["cy"]
    if rank >= MIN_RANK:
        density = grid_find(x, y, d_grid, d_bound_x, d_bound_y)
        if density < target:
            return 0
        return (density - target) / density
    return 0
edges = filter_dropout(edges, d_dropout)

def dropout(row):
    rank = row["rank"]
    if rank >= MIN_RANK:
        rank = MIN_RANK
    dist = row["dist"]
    threshold = -0.0333333 * rank + 0.933333
    if dist < threshold:
        return 0
    return (1 / (1 - threshold)) * (dist - threshold)
edges = filter_dropout(edges, dropout)

# graph = graph.edge_subgraph(edges.index).copy()
# #truncate.largest_component
# graph = ox.utils_graph.get_largest_component(graph, strongly=False)
# edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)

width_map = [
    1,
    1,
    0.8,
    0.8,
    0.6,
    0.6,
    0.4,
    0.4,
    0.4,
    0.4,
    0.4,
    0.4,
    0.4,
]

ig, ax = plt.subplots()
ax.axis("off")
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)

for i in range(len(width_map)):
    subset = edges[edges["rank"] == i+1]
    subset.plot(ax=ax, linewidth=width_map[i], color="black")

plt.show()