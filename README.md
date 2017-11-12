# Scientific-summarization
This repository has code for the reference scope identification task that I was part of, during my time as an intern at NLP Lab, IIT (BHU) in 2017. More details about this task can be found in our paper- [Link will be uploaded soon].

Usage:

1. First run the file "get_features.py" to extract all the features from the reference-citance pairs, writing them into the file 'features.csv' and 'test_features.csv' along with their corresponding labels.

2. Run the file "firstapproach.py" to train and test the CNN model on the above extracted features. Also, you can uncomment the last few lines to use the boosting classifiers simultaneously.

3. Training and test data can be found in respectively named folders
