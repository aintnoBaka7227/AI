import numpy as np  
import pandas as pd  
import sys  

STUDENT_ID = 'a1873825'
DEGREE = 'UG'  

class KdNode:
    def __init__(self, P, D, val):
        self.point = P
        self.dimension = D
        self.value = val
        self.left = None  
        self.right = None  

def FindMedian(P, d):
    sorted_idxs = P[:, d].argsort()
    median_idx = len(P) // 2
    median_p = P[sorted_idxs[median_idx]]
    val = median_p[d]
    return median_p, val

def BuildKdTree(P, D):
    if len(P) == 0:
        return None
    
    if len(P) == 1:
        d = D % 11
        val = P[0][d]
        leaf = KdNode(P[0], d, val)
        # print(f"Leaf node created: dim={d}, val={val}, point={P[0]}")
        return leaf
    
    d = D % 11
    # print(f"\nBuilding tree at depth {D}, dimension {d}")
    # print(f"Number of points: {len(P)}")
    
    median_p, val = FindMedian(P, d)
    # print(f"Median point: {median_p}, value: {val}")
    
    new_node = KdNode(median_p, d, val)
    
    left_pts = []
    right_pts = []
    
    for pt in P:
        if np.array_equal(pt, median_p):
            continue
        if pt[d] <= val:
            left_pts.append(pt)
        else:
            right_pts.append(pt)
    
    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)
    
    # print(f"Left points: {len(left_pts)}, Right points: {len(right_pts)}")
    
    new_node.left = BuildKdTree(left_pts, D + 1)
    new_node.right = BuildKdTree(right_pts, D + 1)
    
    return new_node

def FindNearestNeighbor(root, query_pt, best_point = None, best_distance = float('inf')):
    if root is None:
        return best_point, best_distance
    
    dist_euclidean = 0
    for j in range(11):
        dist_euclidean += (root.point[j] - query_pt[j]) ** 2
    dist_euclidean = np.sqrt(dist_euclidean)
    
    
    
    if dist_euclidean < best_distance:
        best_distance = dist_euclidean
        best_point = root.point
        
    
    if query_pt[root.dimension] <= root.value:
        best_point, best_distance = FindNearestNeighbor(root.left, query_pt, best_point, best_distance)
        
        dist_lb = abs(query_pt[root.dimension] - root.value)
        
        if dist_lb < best_distance:
            best_point, best_distance = FindNearestNeighbor(root.right, query_pt, best_point, best_distance)
    else:
        best_point, best_distance = FindNearestNeighbor(root.right, query_pt, best_point, best_distance)
    
        dist_lb = abs(query_pt[root.dimension] - root.value)
    
        if dist_lb < best_distance:
            best_point, best_distance = FindNearestNeighbor(root.left, query_pt, best_point, best_distance)
    
    return best_point, best_distance

def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    input_dimension = int(sys.argv[3])
    
    train = pd.read_fwf(train_file)
    test = pd.read_fwf(test_file)
    
    train_P = train.values
    test_P = test.values
    
    root = BuildKdTree(train_P, input_dimension)
    
    left_count = 0
    right_count = 0
    
    for p in train_P:
        if np.array_equal(p, root.point):
            continue
        if p[input_dimension] <= root.value:
            left_count += 1
        else:
            right_count += 1
    
    print('.' * input_dimension + 'l' + str(left_count))
    print('.' * input_dimension + 'r' + str(right_count))
    
    for test_pt in test_P:
        best_point,_ = FindNearestNeighbor(root, test_pt)
        if best_point is not None:
            print(int(best_point[-1]))
 

if __name__ == "__main__":
    main()
