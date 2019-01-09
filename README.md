# Sentiment-Analysis
Sentiment Analysis using Multinomial Naive Bayes Classifier

Training_Model.py contains the classifier model. The training dataset was split in 80:20 ratio of Training:Testing. Classification Report is included stating the 'Precision' , 'Recall' and 'f1-score'

Final_Model.py contains the classifier model trained over the complete dataset. The model was saved and is included named as 'MNBC.pkl'.

Prediction_Model.py uses the above saved model 'MNBC.pkl' to predict the values in the test dataset.

Prediction_Results.txt contains the reviews from the test dataset followed by the prediction done by the model.





* The above models were developed using Python 3.6.6 in Spyder 3.3.1 over Anaconda Distribution on Ubuntu 18.10
