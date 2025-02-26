from argparse import ArgumentParser


import word2vec
import numpy as np
import nltk


parser = ArgumentParser()
parser.add_argument('--train', action='store_true',
                    help='Set this flag to train word2vec model')
parser.add_argument('--corpus-path', type=str, default='all.txt',
                    help='Text file for training')
parser.add_argument('--model-path', type=str, default='model.bin',
                    help='Path to save word2vec model')
parser.add_argument('--plot-num', type=int, default=500,
                    help='Number of words to perform dimensionality reduction')
args = parser.parse_args()


if args.train:
    # DEFINE your parameters for training
    SAMPLE = 1e-3
    HIERARCHICAL_SOFTMAX = 1
    MIN_COUNT = 5
    WORDVEC_DIM = 200
    WINDOW = 7
    NEGATIVE_SAMPLES = 0
    ITERATIONS = 50000
    MODEL = 0
    LEARNING_RATE = 0.1

    # train model
    word2vec.word2vec(
        train=args.corpus_path,
        output=args.model_path,
        sample=SAMPLE,
        hs=HIERARCHICAL_SOFTMAX,
        cbow=MODEL,
        size=WORDVEC_DIM,
        min_count=MIN_COUNT,
        window=WINDOW,
        negative=NEGATIVE_SAMPLES,
        iter_=ITERATIONS,
        alpha=LEARNING_RATE,
        verbose=True)
   
    model = word2vec.load(args.model_path)

    indexes, metrics = model.analogy(pos=['Potter', 'Ron'], neg=['Weasley'], n=50)
    for word, similarity in model.generate_response(indexes, metrics):
        print(word, " ", similarity)

else:
    # load model for plotting
    model = word2vec.load(args.model_path)

    vocabs = []                 
    vecs = []                   
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)[:args.plot_num]
    vocabs = vocabs[:args.plot_num]

    '''
    Dimensionality Reduction
    '''
    # from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vecs)


    '''
    Plotting
    '''
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    # JJ: adjective or numeral, ordinal
    # NNP: noun, proper, singular
    # NN: noun, common, singular or mass
    # NNS: noun, common, plural
    # filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    
    
    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    plt.savefig('3.png', dpi=600)
    plt.show()