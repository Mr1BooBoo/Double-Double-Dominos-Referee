# Double Doubble Dominos Referee using Computer Vision SIFT and ORB keypoints
This project leverages computer vision techniques to automate the role of a referee in the game of double-double dominos. The primary goal is to align images of dominos with a template, detect key points and features using SIFT and ORB algorithms, and then determine the best matches for scoring the game.

Please check the instructions below on how to run the program, it is relatively simple.  
First, to align the provided images, I used Orb key points and various openCV algorithms, which were inspired mainly by the labs and partially from the openCV official documentation.  
Then I wrote a function that searches for jpg/jpeg images inside a given directory and applies the sift transformation.  
After the alignment of all images, a function finds half of the dominos inside an image using template matching, once a template was found (half a domino) the coordinates of that template are saved and covered to allow the algorithm to find new pieces and take in consideration already discovered ones.  
The algorithm is constructed such that it starts with step one, which in theory should have one single domino piece.  
And moves on to the next images of the match where it again uses the stored information about previously found dominos to hide them again, therefore the algorithm will always find 2 halves of a domino no matter in which step he finds itself in.  
Lastly, the information about the domino is divided by the theoretical height and width of a cell to find in which row/column it exists.

1. unzip the project file  
2. Please place the testing images inside the regular_tasks folder, and make sure that the test images have a name with a similar format as in the training images (ex. 1_01)  
3. open the Main.py file and please change the urls to match with the location of the project  
4. go to the regular_tasks folder  
5. go to the Aligned folder  
6. if the program runs as intended there should be the aligned images along side the text file which contains both sides of the domino's location and their value. And at the end the score (just like the training format)  
