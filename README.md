
#base_gray,curr_gray: 8bit grayscale images
def get_H_use_OpticalFlowPyrLK(base_gray,curr_gray):


    # find the coordinates of good features to track in base 
    base_features = cv2.goodFeaturesToTrack(base_gray, 3000, .01, 10) 

    # find corresponding features in current photo 
    curr_features = np.array([]) 
    curr_features, pyr_stati, _ = cv2.calcOpticalFlowPyrLK(base_gray, curr_gray, base_features, curr_features, flags=1) 

    tmp_features = np.array([]) 
    tmp_features, pyr_stati, _ = cv2.calcOpticalFlowPyrLK(curr_gray, base_gray, curr_features, tmp_features, flags=1) 

    #Compute the difference of the feature points 
    diff_feature_points  = np.abs(tmp_features - base_features).reshape(-1, 2).max(-1)
    # threshold and keep only the good points 
    d = diff_feature_points < 1

    good_points_ratio = np.sum(d)*100/(len(d) + 0.0000001)

    # only add features for which a match was found to the pruned arrays 
    base_features_pruned = [] 
    curr_features_pruned = [] 
    for index, status in enumerate(pyr_stati): 
        if status == 1: 
         base_features_pruned.append(base_features[index]) 
         curr_features_pruned.append(curr_features[index]) 

    # convert lists to numpy arrays so they can be passed to opencv function 
    bf_final = np.asarray(base_features_pruned) 
    cf_final = np.asarray(curr_features_pruned) 

    if len(bf_final) > 8 and good_points_ratio > 20:
        # find perspective transformation using the arrays of corresponding points 
        transformation, hom_stati = cv2.findHomography(cf_final, bf_final, method=cv2.RANSAC, ransacReprojThreshold=1) 
    else:
        transformation = np.eye(3)

    # transform the images and overlay them to see if they align properly 
    # not what I do in the actual program, just for use in the example code 
    # so that you can see how they align, if you decide to run it 
    #height, width = curr.shape[:2] 
    #mod_photo = cv2.warpPerspective(curr, transformation, (width, height)) 
    
    return transformation
