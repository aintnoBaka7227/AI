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
    median_idx = (len(P) - 1) // 2
    median_p = P[sorted_idxs[median_idx]]
    median_val = median_p[d]
    
    return median_p, median_val

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
        if pt[d] > val:
            right_pts.append(pt)
        else:
            left_pts.append(pt)
    
    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)
    
    # print(f"Left points: {len(left_pts)}, Right points: {len(right_pts)}")
    
    new_node.left = BuildKdTree(left_pts, D + 1)
    new_node.right = BuildKdTree(right_pts, D + 1)
    
    return new_node

def SearchOneNN(root_node, query_pt, best_pt = None, best_dist = float('inf')):
    if root_node is None:
        return best_pt, best_dist
    
    dist_euclidean = 0
    for j in range(11):
        dist_euclidean += (float(root_node.point[j]) - float(query_pt[j])) ** 2
    dist_euclidean = np.sqrt(dist_euclidean)
    
    
    
    if dist_euclidean < best_dist:
        best_dist = dist_euclidean
        best_pt = root_node.point
        
    
    if  float(root_node.value) >= float(query_pt[root_node.dimension]):
        best_pt, best_dist = SearchOneNN(root_node.left, query_pt, best_pt, best_dist)
        
        dist_lb = abs(float(query_pt[root_node.dimension]) - float(root_node.value))
        
        if best_dist > dist_lb:
            best_pt, best_dist = SearchOneNN(root_node.right, query_pt, best_pt, best_dist)
    else:
        best_pt, best_dist = SearchOneNN(root_node.right, query_pt, best_pt, best_dist)
    
        dist_lb = abs(float(query_pt[root_node.dimension]) - float(root_node.value))
    
        if best_dist > dist_lb:
            best_pt, best_dist = SearchOneNN(root_node.left, query_pt, best_pt, best_dist)
    
    return best_pt, best_dist

def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    input_dimension = int(sys.argv[3])
    scaled_dimension = input_dimension % 11
    
    # Read data and skip header row
    train = pd.read_fwf(train_file, skiprows=1, header=None)
    test = pd.read_fwf(test_file, skiprows=1, header=None)
    
    # Convert to float arrays
    train_P = train.values.astype(float)
    test_P = test.values.astype(float)
    
    root_node = BuildKdTree(train_P, scaled_dimension)
    
    left_subtree_count = 0
    right_subtree_count = 0
    
    for p in train_P:
        if np.array_equal(p, root_node.point):
            continue
        if p[scaled_dimension] <= root_node.value:
            left_subtree_count += 1
        else:
            right_subtree_count += 1
    
    print('.' * input_dimension + 'l' + str(left_subtree_count))
    print('.' * input_dimension + 'r' + str(right_subtree_count))
    
    for test_pt in test_P:
        best_pt, _ = SearchOneNN(root_node, test_pt)
        if best_pt is not None:
            print(int(best_pt[-1]))
 

if __name__ == "__main__":
    main()
