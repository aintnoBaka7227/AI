STUDENT_ID = 'a1873825'
DEGREE = 'UG'

import sys
import math
from collections import deque
import heapq

def bfs_func(rows, cols, start, end, grid):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    q = deque([(start, [start])])
    visited = set()
    
    while q: 
        curr, path = q.popleft()
        r, c = curr
        
        if curr == end:
            return path
        
        if curr in visited:
            continue
        visited.add(curr)
        
        for dr, dc in dirs: 
            nr, nc = r + dr, c + dc
            if 1 <= nr <= rows and 1 <= nc <= cols:
                if grid[nr - 1][nc - 1] != 'X' and (nr, nc) not in visited:
                    q.append(((nr, nc), path + [(nr, nc)]))
                    
    return None   

def ucs_func(rows, cols, start, end, grid):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    
    pq = []
    count = 0
    parent = {start: None}
    cost_tracker = {start: 0}
    heapq.heappush(pq, (0, count, start))
    while pq:
        cost, _, (r, c) = heapq.heappop(pq)
        if (r, c) == end:
            return reconstruct(parent, start, end)
        
        for dr, dc in dirs:
            neighbor = (r + dr, c + dc)
            if 1 <= neighbor[0] <= rows and 1 <= neighbor[1] <= cols:
                if grid[neighbor[0] - 1][neighbor[1] - 1] != 'X':
                    elevation_gap = int(grid[neighbor[0] - 1][neighbor[1] - 1]) - int(grid[r - 1][c - 1])
                    new_cost = cost + 1 + max(0, elevation_gap)
                    if neighbor not in cost_tracker or new_cost < cost_tracker[neighbor]:
                        cost_tracker[neighbor] = new_cost
                        count += 1
                        heapq.heappush(pq, (new_cost, count, neighbor))
                        parent[neighbor] = (r, c)
    return None

def reconstruct(came_from, start, end):
    path = []
    current = end
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path
    
def astar_func(rows, cols, start, end, grid, heuristic):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    pq = [] 
    count = 0
    heapq.heappush(pq, (0 + heuristic(start, end), count, 0, start))
    parent = {start: None}
    cost_tracker = {start: 0}
    
    while pq:
        _, _, current_cost, (r, c) = heapq.heappop(pq)
        if (r, c) == end:
            return reconstruct(parent, start, end) 
        
        for dr, dc in dirs:
            neighbor = (r + dr, c + dc)
            if 1 <= neighbor[0] <= rows and 1 <= neighbor[1] <= cols:
                if grid[neighbor[0] - 1][neighbor[1] - 1] != 'X':  
                    elevation_gap = int(grid[neighbor[0] - 1][neighbor[1] - 1]) - int(grid[r - 1][c - 1])
                    new_cost = current_cost + 1 + max(0, elevation_gap)
                    if neighbor not in cost_tracker or new_cost < cost_tracker[neighbor]:
                        cost_tracker[neighbor] = new_cost
                        priority = new_cost + heuristic(neighbor, end)
                        count += 1
                        heapq.heappush(pq, (priority, count, new_cost, neighbor))
                        parent[neighbor] = (r, c)
    return None

def manhattan(r, c):
    return abs(r[0] - c[0]) + abs(r[1] - c[1])
    
def euclidean(r, c):
    return math.sqrt((r[0] - c[0]) ** 2 + (r[1] - c[1]) ** 2)
    
def main():
    mode = sys.argv[1]
    input_file = sys.argv[2]
    algo = sys.argv[3]
    heuristic = sys.argv[4] if len(sys.argv) > 4 else None
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    rows, cols = map(int, lines[0].strip().split())
    start = tuple(map(int, lines[1].strip().split()))
    end = tuple(map(int, lines[2].strip().split()))
    grid = []
    for line in lines[3:]:
        grid.append(line.strip().split())
    
    if algo == 'bfs':
        path = bfs_func(rows, cols, start, end, grid)
    elif algo == 'ucs':
        path = ucs_func(rows, cols, start, end, grid)
    else:
        heuristic_function = manhattan if heuristic == 'manhattan' else euclidean
        path = astar_func(rows, cols, start, end, grid, heuristic_function)
        
    output_grid = []
    for row in grid:
        output_grid.append(row[:])
        
    if path:
        for i, j in path:
            output_grid[i - 1][j - 1] = '*'      
    
    if path is None:
        if mode == 'debug':
            print("path:\nnull")
            print("#visits:\n...")
            print("first visit:\n...")
            print("last visit:\n...")
        else:
            print("null") 
    else:
        if mode == 'release':
            for row in output_grid:
                print(" ".join(row))
        else:
            print("implement later")

if __name__ == "__main__":
    main()
