from __future__ import print_function
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import copy as copy
import random


#This AI model follows encoder - decoder architecture using Attention
#Encoder
#Encoder essentially encodes the input language embeddings into a GRU architecture and
#outputs the GRU output and hidden state data for each words in the input sentence.
class Trans_encoder(nn.Module):
    def __init__(self,input_size, hidden_size,dropout):
        super(Trans_encoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedded = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size,dropout=dropout,bidirectional=True)

    def forward(self, input,input_lengths, hidden=None):
        embedded = self.embedded(input)#input dimensions (max_length of words,batchsize)
        #In pytorch if RNN is used in batches then it needs to be pack_padded before passing through RNN layer
        padded_input= nn.utils.rnn.pack_padded_sequence(embedded,input_lengths,enforce_sorted=False)#embedded dimensions(max_length of words,batchsize,hidden_size)
        #output of GRU shape (seq_len, batch, num_directions * hidden_size)
        #hidden(num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(padded_input,hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)#output dimension (max_length of words,batchsize,2*hidden_size)--2*hidden_size becoz GRU is bi directional
        #summing up the gru's bidirectional hidden states
        output= output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]
        return output, hidden

#Decoder
#Decoder here follows a seq to seq architecture where output of each word that passes through an decoder is given as the input
#to the next step of decoder.
#Attention
#Attention scores are calculated to inform the model which words of the input sentence GRU outputs needs to be attended,
#while predicting words while replying to the chat.

class Trans_attnDecoder(nn.Module):
    def __init__(self,hidden_size, output_size, drop_out):
        super(Trans_attnDecoder,self).__init__()
        self.hidden_size =hidden_size
        self.output_size = output_size
        self.drop_out = drop_out
        self.dropout = nn.Dropout(self.drop_out)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_connec = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_fc = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.gru  = nn.GRU(self.hidden_size,self.hidden_size)
        self.decoder_fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outs):#encoder out dimension max_length of words,batchsize,hidden_size
        embedded = self.embedding(input)#input dimension (1, batch_size)
        embedded = self.dropout(embedded)#embedded dimension(1,batch_size,hiddensize)
        gru_out, gru_hidden = self.gru(embedded,hidden)#gru_out dimension(1,batch_size,hiddensize)

        attn = self.attn_connec(encoder_outs)#attn  dimension max_length of words,batchsize,hidden_size
        sum_attn = t.sum(attn*gru_out,dim=2)#sum_attn dimension max_length,batchsize
        sum_attn= sum_attn.t()# dimension batchsize,max_length
        attn_weights = F.softmax(sum_attn,dim =1).unsqueeze(1) # dimension batchsize, 1, maxlength
        attn_combined = attn_weights.bmm(encoder_outs.transpose(0,1)) # dimension batch_size, hidden_size
        attn_apply = t.cat((gru_out.squeeze(0),attn_combined.squeeze(1)),1)#dimension batch_size, 2* hidden_size
        output = t.tanh(self.attn_fc(attn_apply).unsqueeze(0))#dimension 1, batch_size, hidden_size
        # output = F.relu(output)
        # gated_out, hidden = self.gru(output,hidden)
        output1 = F.softmax(self.decoder_fc(output).squeeze(0),dim=1)#dimension  batch_size, output vocabulary size
        return output1, hidden, attn_weights


class Brain():
    def __init__(self,input_size, hidden_size, output_size, drop_out, batch_size):
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.teacher_forcing_ratio = 0.5

            self.EOS_token = 2
            self.SOS_token =1
            self.PAD_token = 0
            self.encoder = Trans_encoder(input_size,hidden_size,drop_out)
            self.decoder = Trans_attnDecoder(hidden_size,output_size,drop_out)
            self.encoder_optim = optim.Adam(self.encoder.parameters(),lr=0.01)
            self.decoder_optim = optim.Adam(self.decoder.parameters(),lr=0.01)


    def prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq] + [self.EOS_token]
        return idxs

    def zeroPadding(self, l):
        return list(itertools.zip_longest(*l, fillvalue=self.PAD_token))

    def binaryMatrix(self, l):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    def prepare_batch_input(self, input_word2ix):
        lengths = t.tensor([len(indexes) for indexes in input_word2ix])
        padList = self.zeroPadding(input_word2ix)
        padVar = t.LongTensor(padList)
        return padVar, lengths

    def prepare_batch_target(self, target_word2ix):
        max_target_len = max([len(indexes) for indexes in target_word2ix])
        padList = self.zeroPadding(target_word2ix)
        mask = self.binaryMatrix(padList)
        mask = t.ByteTensor(mask)
        padVar = t.LongTensor(padList)
        return padVar, mask, max_target_len

    def maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        target_mod =target.view(-1, 1)
        gather =t.gather(inp, 1, target_mod)
        crossEntropy = -t.log(gather.squeeze(1))
        loss = crossEntropy.masked_select(t.tensor(mask.clone().detach(),dtype=t.bool)).mean()
        return loss, nTotal.item()



    def learniter(self, input_train,target_train, inlang_word2ix,iterations, cal_loss_every):
        iter_count = 0
        iter_loss = 0
        batch_count = 0
        input_batch =[]
        output_batch = []
        max_length =0
        for iter in range(iterations):
            batch_loss = 0
            no_of_batches =0
            loss=0

            for i in range(len(input_train)):
                batch_count += 1
#this call to prepare_sequence function will tokenize the input sentence
                input_word2ix = self.prepare_sequence(str(input_train[i]).split(), inlang_word2ix)
                input_word2ix = t.tensor(input_word2ix, dtype=t.long)
                target_word2ix = self.prepare_sequence(str(target_train[i]).split(), inlang_word2ix)
                target_word2ix = t.tensor(target_word2ix, dtype=t.long)
                input_batch.append(input_word2ix)
                output_batch.append(target_word2ix)

#This below function will create group the input sentences as batches
                if batch_count == self.batch_size:
                    pair_batch = [input_batch,output_batch]
                    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
                    input_batch , output_batch = pair_batch
                    padVar_input, lengths = self.prepare_batch_input(input_batch)
                    padVar_target, mask, max_target_len = self.prepare_batch_target(output_batch)

                    batch_count = 0
                    no_of_batches += 1
                    input_batch = []
                    output_batch = []
                    loss = self.learn( padVar_input, lengths,padVar_target, mask, max_target_len)



                # hidden = t.zeros(1, 1, self.hidden_size)


                batch_loss += loss
            iter_loss += (batch_loss/no_of_batches)
            if iter - iter_count == cal_loss_every:
                print(iter_loss.item()/cal_loss_every)
                iter_count = iter
                iter_loss = 0


    def learn(self, padVar_input, lengths,padVar_target, mask, max_target_len):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
#        encoder_outs = t.zeros(self.max_length, self.hidden_size)
#        encoder_hidden = hidden
        loss =0
        print_losses=[]
        n_total = 0
        encoder_out, encoder_hidden = self.encoder(padVar_input,lengths)#padvar_input dimensions (max_length of words,batchsize)
            # encoder_outs[each_ix] = encoder_out[0,0]
        decoder_hidden = encoder_hidden[:1] #encoder_hidden dim(2,batch_size, hidden)
        decoder_input = t.LongTensor([self.SOS_token for _ in range(self.batch_size)]).unsqueeze(0)
        #Teacher forcing is giving the actual output words in the training set as the input into the decoder instead of the words predicted by the model,
        #this helps the model to converge quickly than not using teaching forcing.
        #But this needs to be given in only for some random sentences or else the model will not learn effectively.
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            for n in range(max_target_len):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input,decoder_hidden,encoder_out)
                #decoder_output shape(batch_size
                target_out = padVar_target[n]
                maskloss , nitems = self.maskNLLLoss(decoder_output, target_out,mask[n])
                loss += maskloss
                decoder_input = target_out.unsqueeze(0)#this line will input the output of the previous decoder word as the input to the next decoder layer
                #forming the basis of the seq to seq architecture.
                print_losses.append(maskloss.item()*nitems)
                n_total += nitems

        else:
            for n in range(max_target_len):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input,decoder_hidden,encoder_out) #encoder_out(max_length,batch_size, hidden_size)
                predicted_output = decoder_output.max(1)[1].detach()
                target_out = padVar_target[n]
                decoder_input = t.LongTensor([[predicted_output[i] for i in range(self.batch_size)]])
                maskloss , nitems = self.maskNLLLoss(decoder_output, target_out,mask[n])
                loss += maskloss
                print_losses.append(maskloss * nitems)
                n_total += nitems

        loss.backward()
        self.encoder_optim.step()
        self.decoder_optim.step()

        return sum(print_losses)/n_total

#For Chatting with the model we are using the beam search decoder , but you can go for the greedy search decoder too.
#but beam search will provide better output as the beam search

    def Beam_search_decoder(self,input_train, inlang_word2ix, beam_size,max_response_words,outix2word):

#Beam search decoder used here, makes use of the recursion for finding the best possible output.
#we can define the beam size here, so for example if the beam size is 3, instead of taking the max probablity words,
#it takes the top 3 maximum probablity words and the same is passed into the seq to seq architecture for all 3 max probablity words as 3 separate decoder architecture
#and subsequentally when predicting for the next word again top 3 probable words are chosen.
# This happens in a recursive fashion until the maximum words in reply limit is reached or EOS token is predicted by the model.


        input_word2ix = self.prepare_sequence(input_train.split(), inlang_word2ix)
        input_word2ix = t.tensor(input_word2ix, dtype=t.long)
        # padVar_input, lengths = self.prepare_batch_input(input_word2ix)
        fulls_sent_ix = self.test_sample(input_word2ix, max_response_words,beam_size)
        max = -100.0000
        max_sent =[]
        for n in fulls_sent_ix:
            sum = 0
            length = len(n)
            # print([k for k in n])
            for v in n:
                sum += t.log(v[0])
                   # print(v[0])
                   # print(v[1])
            # avg = sum/length
            if sum > max:
              max = sum
              max_sent = [h[1] for h in n]
        response = [outix2word[word.item()] for word in max_sent]
        return response

#This tree function is called in a recursive fashion to obtain the desired output.
    def tree(self, max_probs, max_probs_ix,temp_decoder_hidden,temp_encoder_out,max_word_length, n, cache,fulls_sent,beam_size):
        fulls_sent = copy.deepcopy(fulls_sent)
        n += 1
        for i in range(len(max_probs)):
            cache.append([max_probs[:,i], max_probs_ix[:,i]])
            if (n == max_word_length) or (max_probs_ix[:,i] == self.EOS_token):
                m = [c for c in cache]
                fulls_sent.append(m)
                cache.pop()
                break
            max_probs_ix_mod = max_probs_ix[:,i].unsqueeze(0)
            decoder_output, decoder_hidden, _ = self.decoder(max_probs_ix_mod,temp_decoder_hidden,temp_encoder_out)
            # print(decoder_output)
            max_prob, max_prob_ix = decoder_output.topk(k=beam_size, dim=1)
            # print(fulls_sent)
            fulls_sent = self.tree(max_prob, max_prob_ix, decoder_hidden,temp_encoder_out,max_word_length, n, cache,fulls_sent,beam_size)
            # print(fulls_sent)
            cache.pop()
        return fulls_sent



    def test_sample(self,padVar_input, max_word_length,beam_size):
        with t.no_grad():
            lengths = t.tensor(len(padVar_input),dtype= t.long).unsqueeze(0)
            padVar_input = t.LongTensor(padVar_input.unsqueeze(0)).transpose(0,1)
            encoder_out, encoder_hidden = self.encoder(padVar_input, lengths)
            decoder_hidden = encoder_hidden[:1]
            decoder_input = t.tensor([self.SOS_token]).unsqueeze(0)
            decoder_words=[]
            fulls_sent_ix = []
            n=0
            cache=[]
            # for n in range(lengths):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(decoder_input,decoder_hidden,encoder_out)
            # print(decoder_output)
            max_probs, max_probs_ix = t.topk(decoder_output,k=beam_size,dim=1)
            fulls_sent_ix = self.tree(max_probs, max_probs_ix,decoder_hidden,encoder_out,max_word_length, n, cache,fulls_sent_ix,beam_size)
            # decoder_inputs.append(decoder_input)
            # decoder_attentions[n] = decoder_attn,
            # if decoder_input.item() == self.EOS_token:
            #     decoder_words.append('<EOS>')
            #     break
            # else:
            #     decoder_words.append(ix2word[decoder_input.item()])

        return fulls_sent_ix

    # this function  helps to save the model.
    def save(self,path):
        t.save(self.encoder.state_dict(), path+'encoder.pth')
        t.save(self.decoder.state_dict(), path+'decoder.pth')
        print('model saved')

    # this function  helps to load the saved model.
    def load(self, path):
        self.encoder.load_state_dict(t.load(path+'encoder.pth'))
        self.encoder.eval()
        self.decoder.load_state_dict(t.load(path + 'decoder.pth'))
        self.decoder.eval()
        print('model load')



































