# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:59:36 2023

@author: bilal
"""

import os
import cv2
import numpy as np
 
def alignImages(im1, im2):
  MAX_FEATURES = 1000
  GOOD_MATCH_PERCENT = 0.15
  #Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
 
  #Detect ORB features and compute descriptors.
  #orb features were presented in the CV course 
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
 
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  # Sort matches by score
  matches = sorted(matches, key = lambda x:x.distance)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
 
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
 
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
 
  return im1Reg, h




def align_all_images_in_path(path):
    
    #first read the unique tempalte image once
    template_image = r'C:\Users\bilal\Desktop\Master\CV\EDITED.jpg'
    template_image = cv2.imread(template_image, cv2.IMREAD_COLOR)
    
    #now iterate over the folder provided for the project 
    for images in os.listdir(path):
        
        #look for images (that end with jpg or jpeg)
        if (images.endswith(".jpeg") or images.endswith(".jpg")):
            
            #read them using cv2 then apply the sift alignment function written above
            to_align_im = os.path.join(path, images)
            to_align_im = cv2.imread(to_align_im, cv2.IMREAD_COLOR)
            
            #call the alignment function and provide it with the to_align image and the template
            imReg, h = alignImages(to_align_im, template_image)
            
            #save the image to disk
            outFilename = os.path.join(path, 'Aligned', images)
            cv2.imwrite(outFilename, imReg)
            #print(images)

images_path = r'C:\Users\bilal\Desktop\Master\CV\regular_tasks'

align_all_images_in_path(images_path)


path = r'C:\Users\bilal\Desktop\Master\CV\regular_tasks\Aligned'
templates_path = r'C:\Users\bilal\Desktop\Master\CV\board+dominos\halfs'

templates = []
for template in os.listdir(templates_path):
    templates.append(os.path.join(templates_path, template))
del template, templates_path

diamonds = {
1 : [(5,'E'),(5,'G'),(5,'I'),(5,'K'),
        (6,'F'),(6,'J'),
        (7,'E'),(7,'K'),
        (9,'E'),(9,'K'),
        (10,'F'),(10,'J'),
        (11,'E'),(11,'G'),(11,'I'),(11,'K')],

2 : [(3,'E'),(3,'K'),
        (4,'F'),(4,'J'),
        (5,'C'),(5,'M'),
        (6,'D'),(6,'L'),
        
        (10,'D'),(10,'L'),
        (11,'C'),(11,'M'),
        (12,'F'),(12,'J'),
        (13,'E'),(13,'K')],


3 : [(1,'H'),
          (2,'C'),(2,'M'),
          (3,'B'),(3,'N'),
          (4,'D'),(4,'L'),
          
          (8,'A'),(8,'O'),
          
          (12,'D'),(12,'L'),
          (13,'B'),(13,'N'),
          (14,'C'),(14,'M'),
          (15,'H')],


4 : [(1,'D'),(1,'L'),
         (2,'F'),(2,'J'),
         (4,'A'),(4,'O'),
         (6,'B'),(6,'N'),
         
         (10,'B'),(10,'N'),
         (12,'A'),(12,'O'),
         (14,'F'),(14,'J'),
         (15,'D'),(15,'L')],

5 : [(1,'A'),(1,'O'),
         (15,'A'),(15,'O')]
}

all_bounding_boxes = []
def match_current_domino(target_image,target_name):
    
    cell_width = target_image.shape[1] // 15
    cell_height = target_image.shape[0] // 15
    current_faces = []
    to_write_info = []
    score = 0
    for i in range(2):
        #get the best matches 
        best_matches = []
        for template_path in templates:
            template = cv2.imread(template_path)
            result = cv2.matchTemplate(target_image, template, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            #store the best matches in the list
            best_matches.append((template_path, max_val, max_loc))
        
        #sort the best matches by the confidence values
        best_matches = sorted(best_matches, key=lambda x: x[1], reverse=True)
        
        #read the image path
        best_match_box = cv2.imread(best_matches[0][0])
        
        h,w = best_match_box.shape[:2]
        
        #get the best match bounding box coordonates
        top_left = best_matches[0][2]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        all_bounding_boxes.append((top_left, bottom_right))
        
        cv2.rectangle(target_image, top_left, bottom_right, (0, 0, 0), cv2.FILLED)
        
        best_match_loc = best_matches[0][2]
        
        cell_x = chr((best_match_loc[0] // cell_width + 1)+64)
        cell_y = best_match_loc[1] // cell_height + 1
        
        current_faces.append(best_matches[0][0][-5])
        
        for key,val_list in diamonds.items():
            if (cell_y,cell_x) in val_list:
                if len(current_faces) == 2 and current_faces[0]==current_faces[1]:
                    score = key*2
                else:
                    score = key
                    
        info = [cell_x, cell_y, best_matches[0][0][-5]]
        to_write_info.append(info)
        
    with open(os.path.join(path, target_name[:-3]+'txt'), 'w') as f:
        f.write(str(to_write_info[0][1]) + str(to_write_info[0][0])+ " " + str(to_write_info[0][2]) + "\n" + 
                str(to_write_info[1][1]) + str(to_write_info[1][0])+ " " + str(to_write_info[1][2]) + "\n" +
                str(score))
        
        print(f"The best match is {best_matches[0][0][-5]} in cell ({cell_y}{cell_x})")
        #return score
        #cv2.imshow('Result', target_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

def check_all_bboxes(target_image,box_history,target_name):
    #first check if there are previous found bounding boxes 
    #if empty simply start the search for the 2 heads of the new domino
    if not box_history:
        print('no previous boxes, continuing....')
        match_current_domino(target_image,target_name)
    else:
        print('found old bounding boxes, applying....')
        for box in box_history:
            cv2.rectangle(target_image, box[0], box[1], (0, 0, 0), cv2.FILLED)
        match_current_domino(target_image,target_name)


#check_all_bboxes(target)

def get_matches_update_history(path):
    
    #intialize a prev_digit with an unrealistic number
    prev_first_digit = -1

    #read images from the aligned folder
    for image_name in os.listdir(path):
        # Get the first digit in the image name
        first_digit = int(image_name.split("_")[0])
    
        # If the first digit has changed, clear the boxes history
        if first_digit != prev_first_digit:
            all_bounding_boxes.clear()
        
        #update the prev_first_digit 
        prev_first_digit = first_digit
        
        # Load the target image
        target = os.path.join(path, image_name)
        target = cv2.imread(target)
        
        # Call the function and pass the target image
        check_all_bboxes(target, all_bounding_boxes, image_name)
    
        # Update the previous first digit
        prev_first_digit = first_digit


path = r'C:\Users\bilal\Desktop\Master\CV\regular_tasks\Aligned'

get_matches_update_history(path)


















