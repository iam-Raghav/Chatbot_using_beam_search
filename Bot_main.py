from __future__ import print_function
import pickle
import pandas as pd
import torch as t
import random
import re
import unicodedata
from Bot_AiBrain import Brain
#from ConvClassifier import Brain
#Hyperparameters
###################################
hidden_size = 256
drop_out = 0
max_response_words = 16
cal_loss_every = 10
epoch = 50
batch_size = 10
beam_size =3

#Defining the workarea path, since we will be cleaning the dataset, saving and loading the model
###################################
work_area_path = 'E:\AI_files\_Chatbot\\'
###################################
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
###################################
# Lowercase, trim, and remove non-letter characters
###################################
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s
###################################
#Saving the Vocabulary so that it can be loaded during testing of the chat bot
###################################
def save_vocab(path,vocab,name):
    with open (path + name+'.obj','wb') as input_file:
        pickle.dump(vocab,input_file)
    print('dictionary '+name+ ' saved')
###################################

#Loading the Vocabulary for chating with the chat bot.
###################################
def load_vocab(path,name):
    with open (path + name+'.obj','rb') as input_file:
        vocab = pickle.load(input_file)
    print('dictionary ' + name + ' loaded')
    return vocab
###################################

Train_or_test =''
Train_or_test = input('>Do you want to train the bot or chat with it>')

#If you want to train the model , this if condition will be satisfied and model will be trained
#and model will be saved.
if Train_or_test == 'train':
    # File Loading
    ###################################
    filename = 'E:\AI_files\_Chatbot\data\clean_movie_dialogues_mod.txt'
    df = pd.read_csv(filename, header=None, encoding="utf-8", sep='\t')
    ###################################
    df = df.dropna()
    df = df.reset_index(drop= True)


    voc_word2ix = {0: "PAD", 1: "SOS", 2: "EOS"}
    voc_ix2word = {0: "PAD", 1: "SOS", 2: "EOS"}
# Creating word dictionary for the dataset words
###################################
    for i in range(len(df.iloc[:, 1])):
        for word in (str(df.iloc[i, 0]).split()):
            if word not in voc_word2ix:
                voc_word2ix[word] = len(voc_word2ix)
                voc_ix2word[len(voc_word2ix)] = word
        for word in (str(df.iloc[i, 1]).split()):
            if word not in voc_word2ix:
                voc_word2ix[word] = len(voc_word2ix)
                voc_ix2word[len(voc_word2ix)] = word
###################################

    print('vocab size ', len(voc_word2ix))

    input_train = df.iloc[:, 0]
    target_train = df.iloc[:, 1]
#Defining the model
###################################
    trans_model = Brain(len(voc_word2ix), hidden_size, len(voc_word2ix), drop_out, batch_size)
    print(len(input_train))
###################################
# Calling the training function
###################################
    trans_model.learniter(input_train, target_train, voc_word2ix, epoch, cal_loss_every)
###################################
#Saving the encoder and decoder model to use it during chat
###################################
    trans_model.save(work_area_path)
    save_vocab(work_area_path,voc_word2ix,'voc_word2ix')
    save_vocab(work_area_path,voc_ix2word,'voc_ix2word')
###################################
#If you want to Chat with the model , this elif condition will be satisfied and model will be loaded
#and will be ready for chat.
elif Train_or_test == 'chat':
    voc_word2ix = load_vocab(work_area_path,'voc_word2ix')
    voc_ix2word = load_vocab(work_area_path,'voc_ix2word')
    trans_model = Brain(len(voc_word2ix), hidden_size, len(voc_word2ix), drop_out, batch_size)
    trans_model.load(work_area_path)
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            response = trans_model.Beam_search_decoder(input_sentence, voc_word2ix, beam_size, max_response_words,
                                                       voc_ix2word)
            # print(response)
            # Format and print response sentence
            response[:] = [x for x in response if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(response))

        except KeyError:
            print("Error: Encountered unknown word.")

# df1=pd.DataFrame()
# out =[]
# for m in range(len(input_test)):
#     out.append([input_test[m], decoder_words[m], target_test[m]])
#     # print('IN ',input_test[m])
#     # print('Predicted ',decoder_words[m])
#     # print('OUT ',target_test[m])
#     print(out)
#     # print(decoder_attentions[m])
# df1=pd.DataFrame(out,columns = ['Input', 'Predicted', 'Actual_output'])
# print(df1)
# df1.to_csv('E:\AI_files\_Translation\data\out.csv', sep='\t', mode='w', index = False)







