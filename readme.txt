EPJ data science
Finding Epic Moments in Live Content throughLearning from Collective Decisions
Code and trained embeddings
1. Code
1.1. twitch_emote_models.py
This file includes the pytorch-version definition of each model described in paper.
1) freq_classifier, CNN_LSTM_popularity_classifier, LSTM_popularity_classifier: Deep learning based model with frequency, visual content, and chat content.
2) Ensemble_model: our proposed model MINT. To remove final attention layer, please re-write the line 227 as follows; final_out = self.fc(out)
3) FB_freq, FB_img, FB_text: Feature-based model with frequency, visual content, and chat content. 
The training and testing process would be similar to standard machine-learning training and testing pytorch implementation.

1.2. embedding_train.py
This file shows the implementation of emote-embedding described in paper.
total_corpus, emoteonly_corpus,textonly_corpus should be following form. 
class MyIter(object):
    def __iter__(self):
        for corpus in corpus_list:
            sentences = # read the corpus (list of sentence)
            for line in sentences:
                yield line
total_corpus should have the vocabulary which covers emoteonly_corpus and textonly_corpus. 
example training:
save_fname_emote = 'emoteonly_model'
save_fname_text = 'textonly_model'
save_fname_intersect = 'final_model'
# following line saves three embedding models 
embedding_train(total_corpus,emoteonly_corpus,textonly_corpus,save_fname_emote,save_fname_text,save_fname_intersect) 


2. Trained embeddings
twitch_emote_embedding_male_top1000.kvmodel and twitch_emote_embedding_all_top1000.kvmodel are trained by code1.2 with male only and all dataset each.
To access it, you can use the python Gensim KeyedVectors class as follows:
v1 = KeyedVectors.load('twitchemoteembeddingmaletop1000.kvmodel')

If you want full dataset used in the paper, you can contact the author by email (hyun78.song@gmail.com) or please visit the github (https://github.com/dscig)
