STUDENT_ID = 'a1873825'
DEGREE = 'UG'

import sys
import math
from collections import deque
import heapq

def bfs_func(rows, cols, start, end, grid):
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    q = deque([(start, [start])])
    visited = set()
    
    visits = [[0 for _ in range(cols)] for _ in range(rows)]
    first_visit = [[-1 for _ in range(cols)] for _ in range(rows)]
    last_visit = [[-1 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'X':
                visits[i][j] = 'X'
                first_visit[i][j] = 'X'
                last_visit[i][j] = 'X'
                
    visit_count = 1
    visits[start[0] - 1][start[1] - 1] += 1
    first_visit[start[0] - 1][start[1] - 1] = visit_count
    last_visit[start[0] - 1][start[1] - 1] = visit_count
    
    while q: 
        curr, path = q.popleft()
        r, c = curr
        
        # if visits[r - 1][c - 1] != 'X':
        #     visits[r - 1][c - 1] +=1
            
        if first_visit[r - 1][c - 1] == -1:
            first_visit[r - 1][c - 1] = visit_count
            
        last_visit[r - 1][c - 1] = visit_count
        visit_count += 1

        if curr == end:
            for i in range(rows):
                for j in range(cols):
                    if visits[i][j] == 0:
                        visits[i][j] = '.'
                    if first_visit[i][j] == -1:
                        first_visit[i][j] = '.'
                    if last_visit[i][j] == -1:
                        last_visit[i][j] = '.'
                    
            return path, visits, first_visit, last_visit
        
        if curr in visited:
            continue
        visited.add(curr)
        
        for dr, dc in movements: 
            neighbor = (r + dr, c + dc)
            if 1 <= neighbor[0] <= rows and 1 <= neighbor[1] <= cols:
                if grid[neighbor[0] - 1][neighbor[1] - 1] != 'X':
                    
                    visits[neighbor[0] - 1][neighbor[1] - 1] +=1
                    if (first_visit[neighbor[0] - 1][neighbor[1] - 1] == -1):
                        first_visit[neighbor[0] - 1][neighbor[1] - 1] = visit_count
                    last_visit[neighbor[0] - 1][neighbor[1] - 1] = visit_count
                    visit_count += 1
                    
                if neighbor not in visited:
                    q.append((neighbor, path + [neighbor]))
                    
    for i in range(rows):
        for j in range(cols):
            if visits[i][j] == 0:
                visits[i][j] = '.'
            if first_visit[i][j] == -1:
                first_visit[i][j] = '.'
            if last_visit[i][j] == -1:
                last_visit[i][j] = '.'          
    return None, None, None, None  

def ucs_func(rows, cols, start, end, grid):
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    pq = []
    index = 0
    predecessor = {start: None}
    min_cost_tracker = {start: 0}
    heapq.heappush(pq, (0, index, start))
    
    visits = [[0 for _ in range(cols)] for _ in range(rows)]
    first_visit = [[-1 for _ in range(cols)] for _ in range(rows)]
    last_visit = [[-1 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'X':
                visits[i][j] = 'X'
                first_visit[i][j] = 'X'
                last_visit[i][j] = 'X'
                
    visit_count = 1
    visits[start[0] - 1][start[1] - 1] += 1
    first_visit[start[0] - 1][start[1] - 1] = visit_count
    last_visit[start[0] - 1][start[1] - 1] = visit_count
    visit_count += 1
    
    while pq:
        cost, _, (r, c) = heapq.heappop(pq)
        
        if first_visit[r - 1][c - 1] == -1:
            first_visit[r - 1][c - 1] = visit_count
            
        last_visit[r - 1][c - 1] = visit_count
        visit_count += 1
        
        
        if (r, c) == end:
            
            for i in range(rows):
                for j in range(cols):
                    if visits[i][j] == 0:
                        visits[i][j] = '.'
                    if first_visit[i][j] == -1:
                        first_visit[i][j] = '.'
                    if last_visit[i][j] == -1:
                        last_visit[i][j] = '.'
            
            path = []
            current = end
            while current != start:
                path.append(current)
                current = predecessor[current]
            path.append(start)
            path.reverse()
            return path, visits, first_visit, last_visit
        
        for dr, dc in movements:
            neighbor = (r + dr, c + dc)
            if 1 <= neighbor[0] <= rows and 1 <= neighbor[1] <= cols:
                if grid[neighbor[0] - 1][neighbor[1] - 1] != 'X':
                    
                    if grid[neighbor[0] - 1][neighbor[1] - 1] != 'X':
                        visits[neighbor[0] - 1][neighbor[1] - 1] +=1
                    if (first_visit[neighbor[0] - 1][neighbor[1] - 1] == -1):
                        first_visit[neighbor[0] - 1][neighbor[1] - 1] = visit_count
                    last_visit[neighbor[0] - 1][neighbor[1] - 1] = visit_count
                    visit_count += 1
                    
                    elevation_gap = int(grid[neighbor[0] - 1][neighbor[1] - 1]) - int(grid[r - 1][c - 1])
                    new_cost = cost + 1 + max(0, elevation_gap)
                    if neighbor not in min_cost_tracker or new_cost < min_cost_tracker[neighbor]:
                        min_cost_tracker[neighbor] = new_cost
                        index += 1
                        heapq.heappush(pq, (new_cost, index, neighbor))
                        predecessor[neighbor] = (r, c)
                        
    return None, None, None, None
   

def manhattan(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    
def euclidean(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
def astar_func(rows, cols, start, end, grid, heuristic):
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    pq = [] 
    index = 0
    heapq.heappush(pq, (0 + heuristic(start, end), 0, index, start))
    predecessor = {start: None}
    min_cost_tracker = {start: 0}
    
    visits = [[0 for _ in range(cols)] for _ in range(rows)]
    first_visit = [[-1 for _ in range(cols)] for _ in range(rows)]
    last_visit = [[-1 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'X':
                visits[i][j] = 'X'
                first_visit[i][j] = 'X'
                last_visit[i][j] = 'X'
                
    visit_count = 1
    visits[start[0] - 1][start[1] - 1] += 1
    first_visit[start[0] - 1][start[1] - 1] = visit_count
    last_visit[start[0] - 1][start[1] - 1] = visit_count
    visit_count += 1
    
    while pq:
        _, current_cost, _, (r, c) = heapq.heappop(pq)
        
        if first_visit[r - 1][c - 1] == -1:
            first_visit[r - 1][c - 1] = visit_count
            
        last_visit[r - 1][c - 1] = visit_count
        visit_count += 1
        
        if (r, c) == end:
            
            for i in range(rows):
                for j in range(cols):
                    if visits[i][j] == 0:
                        visits[i][j] = '.'
                    if first_visit[i][j] == -1:
                        first_visit[i][j] = '.'
                    if last_visit[i][j] == -1:
                        last_visit[i][j] = '.'
            
            path = []
            current = end
            while current != start:
                path.append(current)
                current = predecessor[current]
            path.append(start)
            path.reverse()
            return path, visits, first_visit, last_visit
        
        for dr, dc in movements:
            neighbor = (r + dr, c + dc)
            if 1 <= neighbor[0] <= rows and 1 <= neighbor[1] <= cols: 
                if grid[neighbor[0] - 1][neighbor[1] - 1] != 'X':  
                    
                    if grid[neighbor[0] - 1][neighbor[1] - 1] != 'X':
                        visits[neighbor[0] - 1][neighbor[1] - 1] +=1
                    if (first_visit[neighbor[0] - 1][neighbor[1] - 1] == -1):
                        first_visit[neighbor[0] - 1][neighbor[1] - 1] = visit_count
                    last_visit[neighbor[0] - 1][neighbor[1] - 1] = visit_count
                    visit_count += 1
                    
                    elevation_gap = int(grid[neighbor[0] - 1][neighbor[1] - 1]) - int(grid[r - 1][c - 1])
                    g = current_cost + 1 + max(0, elevation_gap)
                    if neighbor not in min_cost_tracker or g < min_cost_tracker[neighbor]:
                        min_cost_tracker[neighbor] = g
                        h = heuristic(neighbor, end)
                        f = g + h
                        index += 1
                        heapq.heappush(pq, (f, g, index, neighbor))
                        predecessor[neighbor] = (r, c)
                        
    return None, None, None, None
    
def main():
    mode = sys.argv[1]
    input_file = sys.argv[2]
    algo = sys.argv[3] 
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    rows, cols = map(int, lines[0].strip().split())
    start = tuple(map(int, lines[1].strip().split()))
    end = tuple(map(int, lines[2].strip().split()))
    grid = []
    for line in lines[3:]:
        grid.append(line.strip().split())
    
    if algo == 'bfs':
        path, visits, first_visit, last_visit = bfs_func(rows, cols, start, end, grid)
    elif algo == 'ucs':
        path, visits, first_visit, last_visit = ucs_func(rows, cols, start, end, grid)
    else:
        heuristic = sys.argv[4]
        heuristic_function = manhattan if heuristic == 'manhattan' else euclidean
        path, visits, first_visit, last_visit = astar_func(rows, cols, start, end, grid, heuristic_function)
        
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
            print("path:")
            for row in output_grid:
                print(" ".join(row))
            print("#visits:")
            for row in visits:
                print(" ".join(str(cell) for cell in row))
            print("first visit:")
            for row in first_visit:
                print(" ".join(str(cell) for cell in row))
            print("last visit:")
            for row in last_visit:
                print(" ".join(str(cell) for cell in row))
                        

if __name__ == "__main__":
    main()
