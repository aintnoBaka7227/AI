import numpy as np  
import pandas as pd  
import sys  

STUDENT_ID = 'a1873825'
DEGREE = 'UG'  

class KdNode:
    # node definition
    def __init__(self, P, D, val):
        self.point = P
        self.dimension = D
        self.value = val
        self.left = None  
        self.right = None  

def FindMedian(P, d):
    # sort without changing order using index to get median
    sorted_idxs = P[:, d].argsort()
    median_idx = (len(P) - 1) // 2
    median_p = P[sorted_idxs[median_idx]]
    median_val = median_p[d]
    
    return median_p, median_val

def BuildKdTree(P, D):
    # no node
    if len(P) == 0:
        return None
    
    # leaf node
    if len(P) == 1:
        d = D % 11
        val = P[0][d]
        leaf = KdNode(P[0], d, val)
        
        return leaf
    
    # middel layers node
    d = D % 11
    
    
    median_p, val = FindMedian(P, d)
    
    
    new_node = KdNode(median_p, d, val)
    
    left_pts = []
    right_pts = []
   
    # odd id -> equal to median point go to left subtree 
    for pt in P:
        # prevent infinite loop cause the median point is used for new node already
        if np.array_equal(pt, median_p):
            continue
        if pt[d] > val:
            right_pts.append(pt)
        else:
            left_pts.append(pt)
    
    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)
    
    
    
    new_node.left = BuildKdTree(left_pts, D + 1)
    new_node.right = BuildKdTree(right_pts, D + 1)
    
    return new_node

def SearchOneNN(root_node, query_pt, best_pt = None, best_dist = float('inf')):
    # leaf then return
    if root_node is None:
        return best_pt, best_dist
    
    # euclidean distance 
    dist_euclidean = 0
    for j in range(11):
        dist_euclidean += (float(root_node.point[j]) - float(query_pt[j])) ** 2
    dist_euclidean = np.sqrt(dist_euclidean)
    
    
    # update best predicted area
    if dist_euclidean < best_dist:
        best_dist = dist_euclidean
        best_pt = root_node.point
        
    # go left if query point is less than root node of the subtree
    if  float(root_node.value) >= float(query_pt[root_node.dimension]):
        best_pt, best_dist = SearchOneNN(root_node.left, query_pt, best_pt, best_dist)
        
        # check the other subtree if lower bound is smaller
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
    # scaled large dimension input
    scaled_dimension = input_dimension % 11
    
    # fix header row issue
    train = pd.read_fwf(train_file, skiprows=1, header=None)
    test = pd.read_fwf(test_file, skiprows=1, header=None)
    
    # float issue from gradescope
    train_P = train.values.astype(float)
    test_P = test.values.astype(float)
    
    root_node = BuildKdTree(train_P, scaled_dimension)
    
    left_subtree_count = 0
    right_subtree_count = 0
    
    # count first split -> can be done with a seperate function to check lower level nodes
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
