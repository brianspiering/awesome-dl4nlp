Awesome Deep Learning for Natural Language Processing (NLP) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
====

Contents
----

- __[Courses](#courses)__  
- __[Tutorials / Demos](#tutorials)__  
- __[Talks / Lectures](#talks)__  
- __[Frameworks](#frameworks)__  
- __[Papers](#papers)__  
- __[Blog Posts](#blog-posts)__
- __[Researchers](#researchers)__  
- __[Datasets](#datasets)__  
- __[Miscellaneous](#miscellaneous)__  
- __[Contributing](#contributing)__  

Courses
----
1. CS224d: Deep Learning for Natural Language Processing from Stanford
	- [Course homepage](http://web.stanford.edu/class/cs224n/) A complete survey of the field with videos, lecture slides, and sample student projects.
	- [Course Lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) Video playlist.
	- [Course notes](https://github.com/stanfordnlp/cs224n-winter17-notes) Probably the best "book" on DL for NLP.

Tutorials
----
1. [Deep Learning for Natural Language Processing (without Magic)](http://www.socher.org/index.php/DeepLearningTutorial/DeepLearningTutorial)
2. [A Primer on Neural Network Models for Natural Language Processing](http://u.cs.biu.ac.il/~yogo/nnlp.pdf) 
3. [Deep Learning for Natural Language Processing: Theory and Practice (Tutorial)](https://www.microsoft.com/en-us/research/publication/deep-learning-for-natural-language-processing-theory-and-practice-tutorial/) 
4. [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/mandelbrot)
5. [Recurrent Neural Networks with Word Embeddings](http://deeplearning.net/tutorial/rnnslu.html)
6. [LSTM Networks for Sentiment Analysis](http://deeplearning.net/tutorial/lstm.html)
7. [TensorFlow demo using the Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
8. [LSTMVis: Visual Analysis for Recurrent Neural Networks](http://lstm.seas.harvard.edu/client/index.html)  

Talks
----
1. Ali Ghodsi's lecture on word2vec [part 1](https://www.youtube.com/watch?v=TsEGsdVJjuA) and [part 2](https://www.youtube.com/watch?v=nuirUEmbaJU)
2. [Richard Socher's talk on sentiment analysis, question answering, and sentence-image embeddings](https://www.youtube.com/watch?v=tdLmf8t4oqM)
3. [Deep Learning, an interactive introduction for NLP-ers](http://www.slideshare.net/roelofp/220115dlmeetup)
4. [Deep Natural Language Understanding](http://videolectures.net/deeplearning2016_cho_language_understanding/) 
5. [Deep Learning Summer School, Montreal 2016](http://videolectures.net/deeplearning2016_montreal/) Includes state-of-art language modeling.

Frameworks
----
1. [TensorFlow](https://www.tensorflow.org/) - A cross-platform, general purpose Deep Learning library with Python and C++ API.
2. [Genism: Topic modeling for humans](https://pypi.python.org/pypi/gensim) - A Python package that includes word2vec and doc2vec implementations.
3. [Google’s original word2vec implementation](https://code.google.com/archive/p/word2vec/)
4. [Deeplearning4j’s NLP framework](http://deeplearning4j.org/nlp) - Java implementation.
5. [deepnl](https://github.com/attardi/deepnl) - A Python library for NLP based on Deep Learning neural network architecture.

Papers
----
1. [Deep or shallow, NLP is breaking out](http://dl.acm.org/citation.cfm?id=2874915) - General overview of how Deep Learning is impacting NLP.
2. [Natural Language Processing from Research at Google](http://research.google.com/pubs/NaturalLanguageProcessing.html) - Not all Deep Learning (but mostly).
2. [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) - The original word2vec paper.
3. [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
4. [Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)
5. [Context Dependent Recurrent Neural Network Language Model](http://www.msr-waypoint.com/pubs/176926/rnn_ctxt.pdf)
6. [Translation Modeling with Bidirectional Recurrent Neural Networks](https://www-i6.informatik.rwth-aachen.de/publications/download/936/SundermeyerMartinAlkhouliTamerWuebkerJoernNeyHermann--TranslationModelingwithBidirectionalRecurrentNeuralNetworks--2014.pdf)
7. [Contextual LSTM (CLSTM) models for Large scale NLP tasks](https://arxiv.org/abs/1602.06291)
8. [LSTM Neural Networks for Language Modeling](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.248.4448&rep=rep1&type=pdf)
9. [Exploring the Limits of Language Modeling](http://arxiv.org/pdf/1602.02410.pdf)
10. [Conversational Contextual Cues](https://arxiv.org/abs/1606.00372) - Models context and participants in conversations.
11. [Sequence to sequence learning with neural networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
12. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf)
13. [Learning Character-level Representations for Part-of-Speech Tagging](http://jmlr.org/proceedings/papers/v32/santos14.pdf)
14. [Representation Learning for Text-level Discourse Parsing](http://www.cc.gatech.edu/~jeisenst/papers/ji-acl-2014.pdf)
15. [Fast and Robust Neural Network Joint Models for Statistical Machine Translation](http://acl2014.org/acl2014/P14-1/pdf/P14-1129.pdf)
16. [Parsing With Compositional Vector Grammars](http://www.socher.org/index.php/Main/ParsingWithCompositionalVectorGrammars)
17. [Smart Reply: Automated Response Suggestion for Email](https://arxiv.org/abs/1606.04870)
18. [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) - State-of-the-art performance in NER with bidirectional LSTM with a sequential conditional random layer and transition-based parsing with stack LSTMs.
19. [GloVe: Global Vectors for Word Representation](http://www-nlp.stanford.edu/pubs/glove.pdf) - A "count-based"/co-occurrence model to learn word embeddings.
20. [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449) - State-of-the-art syntactic constituency parsing using generic sequence-to-sequence approach.
22. Skip-Thought Vectors - "unsupervised learning of a generic, distributed sentence encoder"
    - [Paper](http://arxiv.org/abs/1506.06726)
    - [Code](https://github.com/ryankiros/skip-thoughts)


Blog Posts
----
1. [the morning paper: The amazing power of word vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/) - Overview of word vectors.
2. [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
3. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
4. [Machine Learning for Emoji Trends](http://instagram-engineering.tumblr.com/post/117889701472/emojineering-part-1-machine-learning-for-emoji)
5. [Teaching Robots to Feel: Emoji & Deep Learning](http://getdango.com/emoji-and-deep-learning.html)
6. [Computational Linguistics and Deep Learning](http://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00239) - Opinion piece on how Deep Learning fits into the broader picture of text processing.


Researchers
----
1. [Christopher Manning](http://nlp.stanford.edu/manning/)
2. [Ali Ghodsi](https://uwaterloo.ca/data-science/)
3. [Richard Socher](http://www.socher.org/)
4. [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html)

Datasets
----
1. [Dataset from "One Billion Word Language Modeling Benchmark"](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz) - Almost 1B words, already pre-processed text.

Miscellaneous
----
1. [word2vec analogy demo](http://deeplearner.fz-qqq.net/)

-----
Contributing
----
Have anything in mind that you think is awesome and would fit in this list? Feel free to send a pull request!

-----
License
----

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Dr. Brian J. Spiering](http://www.linkedin.com/in/brianspiering/) has waived all copyright and related or neighboring rights to this work.
