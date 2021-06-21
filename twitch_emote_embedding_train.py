import copy
def embedding_train(total_corpus,emoteonly_corpus,textonly_corpus,save_fname_emote,save_fname_text,save_fname_intersect):
    """
    total_corpus, emoteonly_corpus, textonly_corpus should yield the text corpus.
    """
    wv_model = Word2Vec(min_count=100,size=100,negative=0.75,sg=0,hs=1,window=60)
    wv_model.build_vocab(sentences=total_corpus())
    wv_model2 = copy.deepcopy(wv_model)
    
    # train emoteonly
    wv_model.train(sentences=emoteonly_corpus(),epochs=10,total_examples=wv_model.corpus_count)
    wv_model.save(save_fname_emote)
    # train_textonly
    wv_model2.train(sentences=textonly_corpus(),epochs=10,total_examples=wv_model.corpus_count)
    wv_model2.save(save_fname_text)
    
    src_model = Word2Vec.load(save_fname_emote)
    dest_model = Word2Vec.load(save_fname_text)
    
    src_model.wv.save_word2vec_format(save_fname_intersect)
    dest_model.intersect_word2vec_format(save_fname_intersect, lockf=1.0, binary=False)

    dest_model.train(sentences=train_corpus(), total_examples=dest_model.corpus_count, epochs=20)
    dest_model.save(save_fname_intersect)
    return 
