"""Frozen real-abstract fixtures for semantic edge calibration.

Abstracts are from the linked primary arXiv records, acquired 2026-09-06.
Labels were fixed before inspecting cosine scores. Development positives share
one research problem, plus Transformer-to-BERT/RoBERTa architectural lineage.
Validation positives share detection, segmentation, speech-recognition, or policy
optimization problems. Cross-family pairs are negative except for the explicitly
excluded ambiguous development pairs. No measured embeddings or scores are stored.
"""

DEVELOPMENT = {
    "papers": [
        (
            "arxiv:1706.03762",
            "Attention Is All You Need",
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks in an encoder-decoder configuration. The best "
            "performing models also connect the encoder and decoder through an attention "
            "mechanism. We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence and "
            "convolutions entirely. Experiments on two machine translation tasks show these "
            "models to be superior in quality while being more parallelizable and requiring "
            "significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 "
            "English-to-German translation task, improving over the existing best results, "
            "including ensembles by over 2 BLEU. On the WMT 2014 English-to-French "
            "translation task, our model establishes a new single-model state-of-the-art "
            "BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction "
            "of the training costs of the best models from the literature. We show that the "
            "Transformer generalizes well to other tasks by applying it successfully to "
            "English constituency parsing both with large and limited training data.",
            "https://arxiv.org/abs/1706.03762",
        ),
        (
            "arxiv:2004.11886",
            "Lite Transformer with Long-Short Range Attention",
            "Transformer has become ubiquitous in natural language processing (e.g., "
            "machine translation, question answering); however, it requires enormous amount "
            "of computations to achieve high performance, which makes it not suitable for "
            "mobile applications that are tightly constrained by the hardware resources and "
            "battery. In this paper, we present an efficient mobile NLP architecture, Lite "
            "Transformer to facilitate deploying mobile NLP applications on edge devices. "
            "The key primitive is the Long-Short Range Attention (LSRA), where one group of "
            "heads specializes in the local context modeling (by convolution) while another "
            "group specializes in the long-distance relationship modeling (by attention). "
            "Such specialization brings consistent improvement over the vanilla transformer "
            "on three well-established language tasks: machine translation, abstractive "
            "summarization, and language modeling. Under constrained resources (500M/100M "
            "MACs), Lite Transformer outperforms transformer on WMT'14 English-French by "
            "1.2/1.7 BLEU, respectively. Lite Transformer reduces the computation of "
            "transformer base model by 2.5x with 0.3 BLEU score degradation. Combining with "
            "pruning and quantization, we further compressed the model size of Lite "
            "Transformer by 18.2x. For language modeling, Lite Transformer achieves 1.8 "
            "lower perplexity than the transformer at around 500M MACs. Notably, Lite "
            "Transformer outperforms the AutoML-based Evolved Transformer by 0.5 higher "
            "BLEU for the mobile NLP setting without the costly architecture search that "
            "requires more than 250 GPU years. Code has been made available at "
            "https://github.com/mit-han-lab/lite-transformer.",
            "https://arxiv.org/abs/2004.11886",
        ),
        (
            "arxiv:1409.0473",
            "Neural Machine Translation by Jointly Learning to Align and Translate",
            "Neural machine translation is a recently proposed approach to machine "
            "translation. Unlike the traditional statistical machine translation, the "
            "neural machine translation aims at building a single neural network that can "
            "be jointly tuned to maximize the translation performance. The models proposed "
            "recently for neural machine translation often belong to a family of "
            "encoder-decoders and consists of an encoder that encodes a source sentence "
            "into a fixed-length vector from which a decoder generates a translation. In "
            "this paper, we conjecture that the use of a fixed-length vector is a "
            "bottleneck in improving the performance of this basic encoder-decoder "
            "architecture, and propose to extend this by allowing a model to automatically "
            "(soft-)search for parts of a source sentence that are relevant to predicting a "
            "target word, without having to form these parts as a hard segment explicitly. "
            "With this new approach, we achieve a translation performance comparable to the "
            "existing state-of-the-art phrase-based system on the task of English-to-French "
            "translation. Furthermore, qualitative analysis reveals that the "
            "(soft-)alignments found by the model agree well with our intuition.",
            "https://arxiv.org/abs/1409.0473",
        ),
        (
            "arxiv:1508.04025",
            "Effective Approaches to Attention-based Neural Machine Translation",
            "An attentional mechanism has lately been used to improve neural machine "
            "translation (NMT) by selectively focusing on parts of the source sentence "
            "during translation. However, there has been little work exploring useful "
            "architectures for attention-based NMT. This paper examines two simple and "
            "effective classes of attentional mechanism: a global approach which always "
            "attends to all source words and a local one that only looks at a subset of "
            "source words at a time. We demonstrate the effectiveness of both approaches "
            "over the WMT translation tasks between English and German in both directions. "
            "With local attention, we achieve a significant gain of 5.0 BLEU points over "
            "non-attentional systems which already incorporate known techniques such as "
            "dropout. Our ensemble model using different attention architectures has "
            "established a new state-of-the-art result in the WMT'15 English to German "
            "translation task with 25.9 BLEU points, an improvement of 1.0 BLEU points over "
            "the existing best system backed by NMT and an n-gram reranker.",
            "https://arxiv.org/abs/1508.04025",
        ),
        (
            "arxiv:1810.04805",
            "BERT: Pre-training of Deep Bidirectional Transformers for Language "
            "Understanding",
            "We introduce a new language representation model called BERT, which stands for "
            "Bidirectional Encoder Representations from Transformers. Unlike recent "
            "language representation models, BERT is designed to pre-train deep "
            "bidirectional representations from unlabeled text by jointly conditioning on "
            "both left and right context in all layers. As a result, the pre-trained BERT "
            "model can be fine-tuned with just one additional output layer to create "
            "state-of-the-art models for a wide range of tasks, such as question answering "
            "and language inference, without substantial task-specific architecture "
            "modifications. BERT is conceptually simple and empirically powerful. It "
            "obtains new state-of-the-art results on eleven natural language processing "
            "tasks, including pushing the GLUE score to 80.5% (7.7% point absolute "
            "improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD "
            "v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and "
            "SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).",
            "https://arxiv.org/abs/1810.04805",
        ),
        (
            "arxiv:1907.11692",
            "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "Language model pretraining has led to significant performance gains but "
            "careful comparison between different approaches is challenging. Training is "
            "computationally expensive, often done on private datasets of different sizes, "
            "and, as we will show, hyperparameter choices have significant impact on the "
            "final results. We present a replication study of BERT pretraining (Devlin et "
            "al., 2019) that carefully measures the impact of many key hyperparameters and "
            "training data size. We find that BERT was significantly undertrained, and can "
            "match or exceed the performance of every model published after it. Our best "
            "model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results "
            "highlight the importance of previously overlooked design choices, and raise "
            "questions about the source of recently reported improvements. We release our "
            "models and code.",
            "https://arxiv.org/abs/1907.11692",
        ),
        (
            "arxiv:2004.04906",
            "Dense Passage Retrieval for Open-Domain Question Answering",
            "Open-domain question answering relies on efficient passage retrieval to select "
            "candidate contexts, where traditional sparse vector space models, such as "
            "TF-IDF or BM25, are the de facto method. In this work, we show that retrieval "
            "can be practically implemented using dense representations alone, where "
            "embeddings are learned from a small number of questions and passages by a "
            "simple dual-encoder framework. When evaluated on a wide range of open-domain "
            "QA datasets, our dense retriever outperforms a strong Lucene-BM25 system "
            "largely by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and "
            "helps our end-to-end QA system establish new state-of-the-art on multiple "
            "open-domain QA benchmarks.",
            "https://arxiv.org/abs/2004.04906",
        ),
        (
            "arxiv:2007.00808",
            "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text "
            "Retrieval",
            "Conducting text retrieval in a dense learned representation space has many "
            "intriguing advantages over sparse retrieval. Yet the effectiveness of dense "
            "retrieval (DR) often requires combination with sparse retrieval. In this "
            "paper, we identify that the main bottleneck is in the training mechanisms, "
            "where the negative instances used in training are not representative of the "
            "irrelevant documents in testing. This paper presents Approximate nearest "
            "neighbor Negative Contrastive Estimation (ANCE), a training mechanism that "
            "constructs negatives from an Approximate Nearest Neighbor (ANN) index of the "
            "corpus, which is parallelly updated with the learning process to select more "
            "realistic negative training instances. This fundamentally resolves the "
            "discrepancy between the data distribution used in the training and testing of "
            "DR. In our experiments, ANCE boosts the BERT-Siamese DR model to outperform "
            "all competitive dense and sparse retrieval baselines. It nearly matches the "
            "accuracy of sparse-retrieval-and-BERT-reranking using dot-product in the "
            "ANCE-learned representation space and provides almost 100x speed-up.",
            "https://arxiv.org/abs/2007.00808",
        ),
        (
            "arxiv:1609.02907",
            "Semi-Supervised Classification with Graph Convolutional Networks",
            "We present a scalable approach for semi-supervised learning on "
            "graph-structured data that is based on an efficient variant of convolutional "
            "neural networks which operate directly on graphs. We motivate the choice of "
            "our convolutional architecture via a localized first-order approximation of "
            "spectral graph convolutions. Our model scales linearly in the number of graph "
            "edges and learns hidden layer representations that encode both local graph "
            "structure and features of nodes. In a number of experiments on citation "
            "networks and on a knowledge graph dataset we demonstrate that our approach "
            "outperforms related methods by a significant margin.",
            "https://arxiv.org/abs/1609.02907",
        ),
        (
            "arxiv:1710.10903",
            "Graph Attention Networks",
            "We present graph attention networks (GATs), novel neural network architectures "
            "that operate on graph-structured data, leveraging masked self-attentional "
            "layers to address the shortcomings of prior methods based on graph "
            "convolutions or their approximations. By stacking layers in which nodes are "
            "able to attend over their neighborhoods' features, we enable (implicitly) "
            "specifying different weights to different nodes in a neighborhood, without "
            "requiring any kind of costly matrix operation (such as inversion) or depending "
            "on knowing the graph structure upfront. In this way, we address several key "
            "challenges of spectral-based graph neural networks simultaneously, and make "
            "our model readily applicable to inductive as well as transductive problems. "
            "Our GAT models have achieved or matched state-of-the-art results across four "
            "established transductive and inductive graph benchmarks: the Cora, Citeseer "
            "and Pubmed citation network datasets, as well as a protein-protein interaction "
            "dataset (wherein test graphs remain unseen during training).",
            "https://arxiv.org/abs/1710.10903",
        ),
        (
            "arxiv:2006.11239",
            "Denoising Diffusion Probabilistic Models",
            "We present high quality image synthesis results using diffusion probabilistic "
            "models, a class of latent variable models inspired by considerations from "
            "nonequilibrium thermodynamics. Our best results are obtained by training on a "
            "weighted variational bound designed according to a novel connection between "
            "diffusion probabilistic models and denoising score matching with Langevin "
            "dynamics, and our models naturally admit a progressive lossy decompression "
            "scheme that can be interpreted as a generalization of autoregressive decoding. "
            "On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and "
            "a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample "
            "quality similar to ProgressiveGAN. Our implementation is available at "
            "https://github.com/hojonathanho/diffusion",
            "https://arxiv.org/abs/2006.11239",
        ),
        (
            "arxiv:2010.02502",
            "Denoising Diffusion Implicit Models",
            "Denoising diffusion probabilistic models (DDPMs) have achieved high quality "
            "image generation without adversarial training, yet they require simulating a "
            "Markov chain for many steps to produce a sample. To accelerate sampling, we "
            "present denoising diffusion implicit models (DDIMs), a more efficient class of "
            "iterative implicit probabilistic models with the same training procedure as "
            "DDPMs. In DDPMs, the generative process is defined as the reverse of a "
            "Markovian diffusion process. We construct a class of non-Markovian diffusion "
            "processes that lead to the same training objective, but whose reverse process "
            "can be much faster to sample from. We empirically demonstrate that DDIMs can "
            "produce high quality samples $10 \\times$ to $50 \\times$ faster in terms of "
            "wall-clock time compared to DDPMs, allow us to trade off computation for "
            "sample quality, and can perform semantically meaningful image interpolation "
            "directly in the latent space.",
            "https://arxiv.org/abs/2010.02502",
        ),
        (
            "arxiv:1512.03385",
            "Deep Residual Learning for Image Recognition",
            "Deeper neural networks are more difficult to train. We present a residual "
            "learning framework to ease the training of networks that are substantially "
            "deeper than those used previously. We explicitly reformulate the layers as "
            "learning residual functions with reference to the layer inputs, instead of "
            "learning unreferenced functions. We provide comprehensive empirical evidence "
            "showing that these residual networks are easier to optimize, and can gain "
            "accuracy from considerably increased depth. On the ImageNet dataset we "
            "evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG "
            "nets but still having lower complexity. An ensemble of these residual nets "
            "achieves 3.57% error on the ImageNet test set. This result won the 1st place "
            "on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 "
            "with 100 and 1000 layers. The depth of representations is of central "
            "importance for many visual recognition tasks. Solely due to our extremely deep "
            "representations, we obtain a 28% relative improvement on the COCO object "
            "detection dataset. Deep residual nets are foundations of our submissions to "
            "ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks "
            "of ImageNet detection, ImageNet localization, COCO detection, and COCO "
            "segmentation.",
            "https://arxiv.org/abs/1512.03385",
        ),
        (
            "arxiv:1603.05027",
            "Identity Mappings in Deep Residual Networks",
            "Deep residual networks have emerged as a family of extremely deep "
            "architectures showing compelling accuracy and nice convergence behaviors. In "
            "this paper, we analyze the propagation formulations behind the residual "
            "building blocks, which suggest that the forward and backward signals can be "
            "directly propagated from one block to any other block, when using identity "
            "mappings as the skip connections and after-addition activation. A series of "
            "ablation experiments support the importance of these identity mappings. This "
            "motivates us to propose a new residual unit, which makes training easier and "
            "improves generalization. We report improved results using a 1001-layer ResNet "
            "on CIFAR-10 (4.62% error) and CIFAR-100, and a 200-layer ResNet on ImageNet. "
            "Code is available at: https://github.com/KaimingHe/resnet-1k-layers",
            "https://arxiv.org/abs/1603.05027",
        ),
    ],
    "positive_pairs": [
        ("arxiv:1706.03762", "arxiv:2004.11886"),
        ("arxiv:1706.03762", "arxiv:1409.0473"),
        ("arxiv:1706.03762", "arxiv:1508.04025"),
        ("arxiv:1706.03762", "arxiv:1810.04805"),
        ("arxiv:1706.03762", "arxiv:1907.11692"),
        ("arxiv:2004.11886", "arxiv:1409.0473"),
        ("arxiv:2004.11886", "arxiv:1508.04025"),
        ("arxiv:1409.0473", "arxiv:1508.04025"),
        ("arxiv:1810.04805", "arxiv:1907.11692"),
        ("arxiv:2004.04906", "arxiv:2007.00808"),
        ("arxiv:1609.02907", "arxiv:1710.10903"),
        ("arxiv:2006.11239", "arxiv:2010.02502"),
        ("arxiv:1512.03385", "arxiv:1603.05027"),
    ],
    "excluded_pairs": [
        ("arxiv:2004.11886", "arxiv:1810.04805"),
        ("arxiv:2004.11886", "arxiv:1907.11692"),
        ("arxiv:1409.0473", "arxiv:1810.04805"),
        ("arxiv:1409.0473", "arxiv:1907.11692"),
        ("arxiv:1508.04025", "arxiv:1810.04805"),
        ("arxiv:1508.04025", "arxiv:1907.11692"),
        ("arxiv:1810.04805", "arxiv:2004.04906"),
        ("arxiv:1810.04805", "arxiv:2007.00808"),
        ("arxiv:1907.11692", "arxiv:2004.04906"),
        ("arxiv:1907.11692", "arxiv:2007.00808"),
    ],
}

VALIDATION = {
    "papers": [
        (
            "arxiv:1506.01497",
            "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
            "State-of-the-art object detection networks depend on region proposal "
            "algorithms to hypothesize object locations. Advances like SPPnet and Fast "
            "R-CNN have reduced the running time of these detection networks, exposing "
            "region proposal computation as a bottleneck. In this work, we introduce a "
            "Region Proposal Network (RPN) that shares full-image convolutional features "
            "with the detection network, thus enabling nearly cost-free region proposals. "
            "An RPN is a fully convolutional network that simultaneously predicts object "
            "bounds and objectness scores at each position. The RPN is trained end-to-end "
            "to generate high-quality region proposals, which are used by Fast R-CNN for "
            "detection. We further merge RPN and Fast R-CNN into a single network by "
            "sharing their convolutional features---using the recently popular terminology "
            "of neural networks with 'attention' mechanisms, the RPN component tells the "
            "unified network where to look. For the very deep VGG-16 model, our detection "
            "system has a frame rate of 5fps (including all steps) on a GPU, while "
            "achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, "
            "and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO "
            "2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place "
            "winning entries in several tracks. Code has been made publicly available.",
            "https://arxiv.org/abs/1506.01497",
        ),
        (
            "arxiv:1708.02002",
            "Focal Loss for Dense Object Detection",
            "The highest accuracy object detectors to date are based on a two-stage "
            "approach popularized by R-CNN, where a classifier is applied to a sparse set "
            "of candidate object locations. In contrast, one-stage detectors that are "
            "applied over a regular, dense sampling of possible object locations have the "
            "potential to be faster and simpler, but have trailed the accuracy of two-stage "
            "detectors thus far. In this paper, we investigate why this is the case. We "
            "discover that the extreme foreground-background class imbalance encountered "
            "during training of dense detectors is the central cause. We propose to address "
            "this class imbalance by reshaping the standard cross entropy loss such that it "
            "down-weights the loss assigned to well-classified examples. Our novel Focal "
            "Loss focuses training on a sparse set of hard examples and prevents the vast "
            "number of easy negatives from overwhelming the detector during training. To "
            "evaluate the effectiveness of our loss, we design and train a simple dense "
            "detector we call RetinaNet. Our results show that when trained with the focal "
            "loss, RetinaNet is able to match the speed of previous one-stage detectors "
            "while surpassing the accuracy of all existing state-of-the-art two-stage "
            "detectors. Code is at: https://github.com/facebookresearch/Detectron.",
            "https://arxiv.org/abs/1708.02002",
        ),
        (
            "arxiv:1506.02640",
            "You Only Look Once: Unified, Real-Time Object Detection",
            "We present YOLO, a new approach to object detection. Prior work on object "
            "detection repurposes classifiers to perform detection. Instead, we frame "
            "object detection as a regression problem to spatially separated bounding boxes "
            "and associated class probabilities. A single neural network predicts bounding "
            "boxes and class probabilities directly from full images in one evaluation. "
            "Since the whole detection pipeline is a single network, it can be optimized "
            "end-to-end directly on detection performance. Our unified architecture is "
            "extremely fast. Our base YOLO model processes images in real-time at 45 frames "
            "per second. A smaller version of the network, Fast YOLO, processes an "
            "astounding 155 frames per second while still achieving double the mAP of other "
            "real-time detectors. Compared to state-of-the-art detection systems, YOLO "
            "makes more localization errors but is far less likely to predict false "
            "detections where nothing exists. Finally, YOLO learns very general "
            "representations of objects. It outperforms all other detection methods, "
            "including DPM and R-CNN, by a wide margin when generalizing from natural "
            "images to artwork on both the Picasso Dataset and the People-Art Dataset.",
            "https://arxiv.org/abs/1506.02640",
        ),
        (
            "arxiv:1411.4038",
            "Fully Convolutional Networks for Semantic Segmentation",
            "Convolutional networks are powerful visual models that yield hierarchies of "
            "features. We show that convolutional networks by themselves, trained "
            "end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic "
            'segmentation. Our key insight is to build "fully convolutional" networks that '
            "take input of arbitrary size and produce correspondingly-sized output with "
            "efficient inference and learning. We define and detail the space of fully "
            "convolutional networks, explain their application to spatially dense "
            "prediction tasks, and draw connections to prior models. We adapt contemporary "
            "classification networks (AlexNet, the VGG net, and GoogLeNet) into fully "
            "convolutional networks and transfer their learned representations by "
            "fine-tuning to the segmentation task. We then define a novel architecture that "
            "combines semantic information from a deep, coarse layer with appearance "
            "information from a shallow, fine layer to produce accurate and detailed "
            "segmentations. Our fully convolutional network achieves state-of-the-art "
            "segmentation of PASCAL VOC (20% relative improvement to 62.2% mean IU on "
            "2012), NYUDv2, and SIFT Flow, while inference takes one third of a second for "
            "a typical image.",
            "https://arxiv.org/abs/1411.4038",
        ),
        (
            "arxiv:1505.04597",
            "U-Net: Convolutional Networks for Biomedical Image Segmentation",
            "There is large consent that successful training of deep networks requires many "
            "thousand annotated training samples. In this paper, we present a network and "
            "training strategy that relies on the strong use of data augmentation to use "
            "the available annotated samples more efficiently. The architecture consists of "
            "a contracting path to capture context and a symmetric expanding path that "
            "enables precise localization. We show that such a network can be trained "
            "end-to-end from very few images and outperforms the prior best method (a "
            "sliding-window convolutional network) on the ISBI challenge for segmentation "
            "of neuronal structures in electron microscopic stacks. Using the same network "
            "trained on transmitted light microscopy images (phase contrast and DIC) we won "
            "the ISBI cell tracking challenge 2015 in these categories by a large margin. "
            "Moreover, the network is fast. Segmentation of a 512x512 image takes less than "
            "a second on a recent GPU. The full implementation (based on Caffe) and the "
            "trained networks are available at "
            "http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net .",
            "https://arxiv.org/abs/1505.04597",
        ),
        (
            "arxiv:1511.00561",
            "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image "
            "Segmentation",
            "We present a novel and practical deep fully convolutional neural network "
            "architecture for semantic pixel-wise segmentation termed SegNet. This core "
            "trainable segmentation engine consists of an encoder network, a corresponding "
            "decoder network followed by a pixel-wise classification layer. The "
            "architecture of the encoder network is topologically identical to the 13 "
            "convolutional layers in the VGG16 network. The role of the decoder network is "
            "to map the low resolution encoder feature maps to full input resolution "
            "feature maps for pixel-wise classification. The novelty of SegNet lies is in "
            "the manner in which the decoder upsamples its lower resolution input feature "
            "map(s). Specifically, the decoder uses pooling indices computed in the "
            "max-pooling step of the corresponding encoder to perform non-linear "
            "upsampling. This eliminates the need for learning to upsample. The upsampled "
            "maps are sparse and are then convolved with trainable filters to produce dense "
            "feature maps. We compare our proposed architecture with the widely adopted FCN "
            "and also with the well known DeepLab-LargeFOV, DeconvNet architectures. This "
            "comparison reveals the memory versus accuracy trade-off involved in achieving "
            "good segmentation performance. SegNet was primarily motivated by scene "
            "understanding applications. Hence, it is designed to be efficient both in "
            "terms of memory and computational time during inference. It is also "
            "significantly smaller in the number of trainable parameters than other "
            "competing architectures. We also performed a controlled benchmark of SegNet "
            "and other architectures on both road scenes and SUN RGB-D indoor scene "
            "segmentation tasks. We show that SegNet provides good performance with "
            "competitive inference time and more efficient inference memory-wise as "
            "compared to other architectures. We also provide a Caffe implementation of "
            "SegNet and a web demo at http://mi.eng.cam.ac.uk/projects/segnet/.",
            "https://arxiv.org/abs/1511.00561",
        ),
        (
            "arxiv:1412.5567",
            "Deep Speech: Scaling up end-to-end speech recognition",
            "We present a state-of-the-art speech recognition system developed using "
            "end-to-end deep learning. Our architecture is significantly simpler than "
            "traditional speech systems, which rely on laboriously engineered processing "
            "pipelines; these traditional systems also tend to perform poorly when used in "
            "noisy environments. In contrast, our system does not need hand-designed "
            "components to model background noise, reverberation, or speaker variation, but "
            "instead directly learns a function that is robust to such effects. We do not "
            'need a phoneme dictionary, nor even the concept of a "phoneme." Key to our '
            "approach is a well-optimized RNN training system that uses multiple GPUs, as "
            "well as a set of novel data synthesis techniques that allow us to efficiently "
            "obtain a large amount of varied data for training. Our system, called Deep "
            "Speech, outperforms previously published results on the widely studied "
            "Switchboard Hub5'00, achieving 16.0% error on the full test set. Deep Speech "
            "also handles challenging noisy environments better than widely used, "
            "state-of-the-art commercial speech systems.",
            "https://arxiv.org/abs/1412.5567",
        ),
        (
            "arxiv:1303.5778",
            "Speech Recognition with Deep Recurrent Neural Networks",
            "Recurrent neural networks (RNNs) are a powerful model for sequential data. "
            "End-to-end training methods such as Connectionist Temporal Classification make "
            "it possible to train RNNs for sequence labelling problems where the "
            "input-output alignment is unknown. The combination of these methods with the "
            "Long Short-term Memory RNN architecture has proved particularly fruitful, "
            "delivering state-of-the-art results in cursive handwriting recognition. "
            "However RNN performance in speech recognition has so far been disappointing, "
            "with better results returned by deep feedforward networks. This paper "
            "investigates \\emph{deep recurrent neural networks}, which combine the "
            "multiple levels of representation that have proved so effective in deep "
            "networks with the flexible use of long range context that empowers RNNs. When "
            "trained end-to-end with suitable regularisation, we find that deep Long "
            "Short-term Memory RNNs achieve a test set error of 17.7% on the TIMIT phoneme "
            "recognition benchmark, which to our knowledge is the best recorded score.",
            "https://arxiv.org/abs/1303.5778",
        ),
        (
            "arxiv:1502.05477",
            "Trust Region Policy Optimization",
            "We describe an iterative procedure for optimizing policies, with guaranteed "
            "monotonic improvement. By making several approximations to the "
            "theoretically-justified procedure, we develop a practical algorithm, called "
            "Trust Region Policy Optimization (TRPO). This algorithm is similar to natural "
            "policy gradient methods and is effective for optimizing large nonlinear "
            "policies such as neural networks. Our experiments demonstrate its robust "
            "performance on a wide variety of tasks: learning simulated robotic swimming, "
            "hopping, and walking gaits; and playing Atari games using images of the screen "
            "as input. Despite its approximations that deviate from the theory, TRPO tends "
            "to give monotonic improvement, with little tuning of hyperparameters.",
            "https://arxiv.org/abs/1502.05477",
        ),
        (
            "arxiv:1707.06347",
            "Proximal Policy Optimization Algorithms",
            "We propose a new family of policy gradient methods for reinforcement learning, "
            "which alternate between sampling data through interaction with the "
            'environment, and optimizing a "surrogate" objective function using stochastic '
            "gradient ascent. Whereas standard policy gradient methods perform one gradient "
            "update per data sample, we propose a novel objective function that enables "
            "multiple epochs of minibatch updates. The new methods, which we call proximal "
            "policy optimization (PPO), have some of the benefits of trust region policy "
            "optimization (TRPO), but they are much simpler to implement, more general, and "
            "have better sample complexity (empirically). Our experiments test PPO on a "
            "collection of benchmark tasks, including simulated robotic locomotion and "
            "Atari game playing, and we show that PPO outperforms other online policy "
            "gradient methods, and overall strikes a favorable balance between sample "
            "complexity, simplicity, and wall-time.",
            "https://arxiv.org/abs/1707.06347",
        ),
    ],
    "positive_pairs": [
        ("arxiv:1506.01497", "arxiv:1708.02002"),
        ("arxiv:1506.01497", "arxiv:1506.02640"),
        ("arxiv:1708.02002", "arxiv:1506.02640"),
        ("arxiv:1411.4038", "arxiv:1505.04597"),
        ("arxiv:1411.4038", "arxiv:1511.00561"),
        ("arxiv:1505.04597", "arxiv:1511.00561"),
        ("arxiv:1412.5567", "arxiv:1303.5778"),
        ("arxiv:1502.05477", "arxiv:1707.06347"),
    ],
    "excluded_pairs": [],
}
