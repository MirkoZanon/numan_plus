import numpy as np

def filter_nn(indices, dist, corr_thresh, signal2, signal_ref, centers_refplane, centers_plane2):
    match_index_array = np.ones((indices.shape[0],))*(-1) #index of cell in plane2 merged with corresponding row cell in refplane. -1 = unique
    for i, ix2 in enumerate(indices):
        if (np.corrcoef(signal2[int(ix2)][np.newaxis],signal_ref[i][np.newaxis])[0,1]>=corr_thresh) & (dist[i]<1):
            match_index_array[i] = ix2
            my_dist = np.sqrt((centers_refplane[i][0]-centers_plane2[int(ix2)][0])**2+(centers_refplane[i][1]-centers_plane2[int(ix2)][1])**2)
    return match_index_array

def nearest_neighbourds(centers_plane2, centers_refplane, radius):
    index_neighbours = np.ones(len(centers_refplane))-2
    distance_neighbours = np.ones(len(centers_refplane))*1000
    for idx, p_ref in enumerate(centers_refplane):
        current_min_dist = radius
        for idx2, p2 in enumerate(centers_plane2):
           dist = np.sqrt((p_ref[0]-p2[0])**2+(p_ref[1]-p2[1])**2)
           if dist<=current_min_dist:
               current_min_dist = dist
               index_neighbours[idx] = idx2
               distance_neighbours[idx] = dist          
    return distance_neighbours, index_neighbours

def merge_units(plane_name_2, plane_name_ref, dist_thresh, corr_thresh, CENTERS_dict, DFF_dict):
    centers_refplane = CENTERS_dict[plane_name_ref]
    centers_plane2 = CENTERS_dict[plane_name_2]
    dff_refplane = DFF_dict[plane_name_ref]
    dff_plane2 = DFF_dict[plane_name_2]
    dist, idx2 = nearest_neighbourds(centers_plane2, centers_refplane, radius = dist_thresh)
    #idx2 -= 1   # analysis.nearest_neighbour_assignmnet starts to number from 1, we adjust back to 0
    merged = filter_nn(idx2, dist, corr_thresh, dff_plane2, dff_refplane, centers_refplane, centers_plane2)
    return merged