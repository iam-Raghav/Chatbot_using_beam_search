# Chatbot_using_beam_search
Building the chatbot using encoder - seq to seq beam search decoder(Attention)


# Pre-requistics
    Python- 3.7
    Torch - 1.0.0
    numpy -1.16.4
    sklearn - 0.21.3
    pandas - 0.25.0
    
# Files
Dataset can be downloaded from the link https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
1. Bot_Text_cleaner.py - This file will clean the dataset by combining the movie_conversations.txt and movie_lines.txt, by removing the stopwords, punctuations and converting the format to UTF-8
2. Bot_main.py - This file contains the main code that needs to be run to predict the french sentence.
3. Bot_AiBrain.py - This file contains the training, testing and Encoder decoder model required for the translation.

The model needs to run for as many iterations as possible, untill the loss function converges.
The response will be stupid if the iterations are very less and also when the model runs for a longer iteration, it might take few days to complete.

Note:- Text cleaner file is made as a separate file for re-usability and saving time to avoid running the text cleaning process for each and every time Translate_main.py file is run.
    
# Downloads and Setup
Once you clone this repo, run the Bot_main.py file to do the sentiment analysis and to train the model.

# Evalution metric
No evalution metric is used in this model.
