STUDENT_ID = 'a1873825'
DEGREE = 'UG'

import sys
import math
from collections import deque
import heapq

def bfs(rows, cols, start, end, grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
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
        
        for dr, dc in directions: 
            nr, nc = r + dr, c + dc
            if 1 <= nr <= rows and 1 <= nc <= cols:
                if grid[nr - 1][nc - 1] != 'X' and (nr, nc) not in visited:
                    q.append(((nr, nc), path + [(nr, nc)]))
                    
    return None   

def ucs(rows, cols, start, end, grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    
    pq = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while pq:
        cost, (x, y) = heapq.heappop(pq)
        if (x, y) == end:
            return reconstruct_path(came_from, start, end)
        
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if 1 <= neighbor[0] <= rows and 1 <= neighbor[1] <= cols:
                if grid[neighbor[0] - 1][neighbor[1] - 1] != 'X':
                    elevation_diff = int(grid[neighbor[0] - 1][neighbor[1] - 1]) - int(grid[x - 1][y - 1])
                    new_cost = cost_so_far[(x, y)] + 1 + max(0, elevation_diff)
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        heapq.heappush(pq, (new_cost, neighbor))
                        came_from[neighbor] = (x, y)
    return None

def reconstruct_path(came_from, start, end):
    path = []
    current = end
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path
    
def astar(rows, cols, start, end, grid, heuristic):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    pq = [(0 + heuristic(start, end), 0, start)] 
    heapq.heapify(pq)
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while pq:
        _, current_cost, (x, y) = heapq.heappop(pq)
        if (x, y) == end:
            return reconstruct_path(came_from, start, end)
        
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if 1 <= neighbor[0] <= rows and 1 <= neighbor[1] <= cols:
                if grid[neighbor[0] - 1][neighbor[1] - 1] != 'X':  
                    elevation_diff = int(grid[neighbor[0] - 1][neighbor[1] - 1]) - int(grid[x - 1][y - 1])
                    new_cost = current_cost + 1 + max(0, elevation_diff)
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + heuristic(neighbor, end)
                        heapq.heappush(pq, (priority, new_cost, neighbor))
                        came_from[neighbor] = (x, y)
    return None

def manhattan(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])
    
def euclidean(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    
def main():
    mode = sys.argv[1]
    input_file = sys.argv[2]
    algorithm = sys.argv[3]
    heuristic = sys.argv[4] if len(sys.argv) > 4 else None
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    rows, cols = map(int, lines[0].strip().split())
    start = tuple(map(int, lines[1].strip().split()))
    goal = tuple(map(int, lines[2].strip().split()))
    grid = [line.strip().split() for line in lines[3:]]
    
    if algorithm == 'bfs':
        path = bfs(rows, cols, start, goal, grid)
    elif algorithm == 'ucs':
        path = ucs(rows, cols, start, goal, grid)
    else:
        heuristic_function = manhattan if heuristic == 'manhattan' else euclidean
        path = astar(rows, cols, start, goal, grid, heuristic_function)
        
    output_grid = [row[:] for row in grid]
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
