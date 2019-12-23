from __future__ import print_function
import csv
import pandas as pd
import re
import unicodedata


rawtext_filename = 'E:\AI_files\_Chatbot\data\movie_lines.txt' #KEY IN PATH OF SOURCE FILE
rawtext_ids = 'E:\AI_files\_Chatbot\data\movie_conversations.txt'
clean_data_pairs = 'E:\AI_files\_Chatbot\data\clean_movie_dialogues.txt'

max_length = 8

#File Loading
###################################

df = pd.read_csv(rawtext_filename,header=None, encoding = "iso-8859-1", sep ='@')
df1= pd.read_csv(rawtext_ids,header=None, encoding = "iso-8859-1", sep ='\t',converters= {3: lambda x: x.strip("[]").replace("'","").split(", ")})
conv_pairs =[]
###################################

#Removing punctuations from the text
###################################
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s) #replaces .!? with a preceding space
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) #removes everything other than a-zA-Z.!?
    s = re.sub(r'\s+', ' ', s) #removes \t \n values
    return s


df.sort_values(by =0,inplace= True)
df = df.dropna()

for i in range(len(df1.iloc[:,1])):
    line_id=[]
    line_id = df1.iloc[i,3]
    for j in range(len(line_id) - 1):
        if (len(normalizeString(str(df.loc[df[0]==line_id[j],4].values)).split()) <= max_length):
            input =normalizeString(str(df.loc[df[0]==line_id[j],4].values)).lstrip()
            output =normalizeString(str(df.loc[df[0]==line_id[j+1],4].values)).lstrip()
            conv_pairs.append([input,output])
with open(clean_data_pairs, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter='\t', lineterminator='\n')
    for pair in conv_pairs:
        writer.writerow(pair)

print("DONE...")


