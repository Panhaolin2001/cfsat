from collections import defaultdict
import random

def build_graph(synerpairs):
    graph = defaultdict(list)
    for u, v in synerpairs:
        graph[u].append(v)
    return graph

def generate_population(synerpairs, size=100, max_length=20, min_length=4):
    graph = build_graph(synerpairs)
    nodes = list(graph.keys())
    population = set()

    while len(population) < size:
        path = []
        head_node = random.choice(nodes)
        path.append(head_node)
        
        current_node = head_node
        while len(path) < max_length:
            next_nodes = graph[current_node]
            if len(next_nodes) != 0:
                next_node = random.choice(next_nodes)
                path.append(next_node)
                current_node = next_node
            else:
                break
        
        if len(path) >= min_length:
            population.add(tuple(path))
    
    return [list(p) for p in population]

def generate_population_nograph(synerpairs, size=100, max_length=20, min_length=4):
    population = set()
    
    while len(population) < size:
        path = []
        
        # Randomly determine the length of the path within the specified range
        path_length = random.randint(min_length, max_length)
        
        # Randomly select elements from synerpairs to build the path
        while len(path) < path_length:
            u, v = random.choice(synerpairs)
            path.append(u)
            if len(path) < path_length: 
                path.append(v)
        
        population.add(tuple(path))
    
    return [list(p) for p in population]