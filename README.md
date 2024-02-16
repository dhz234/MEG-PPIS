## MEGPPIS-a fast protein-protein interaction sites prediction method based on multi-scale graph information and equivariant graph neural network

Protein-protein interaction sites is crucial for deciphering protein action mechanisms and related medical research, which is the key issue in protein action research. Recent studies have shown that graph neural networks have achieved outstanding performance in predicting PPIS. Nevertheless, these studies often neglects the modeling of the inherent presence of diverse scales on the graph and symmetries  in protein molecules within three-dimensional space. In response to this gap, this paper proposes so-called MEG-PPIS approach, a PPIS prediction method based on multi-scale graph information and E (3) equivariant graph neural network (EGNN). There are two channels in MEG-PPIS: the original graph and the subgraph obtained by graph pooling. The model can iteratively update the features of the original graph and subgraph through the weight-sharing EGNN. Subsequently, the max-pooling operation aggregates the updated features of the original graph and subgraph. Ultimately, the model feeds node features into the prediction layer to generate the prediction results. Comparative assessments against other advanced methods on benchmark datasets reveal that MEG-PPIS achieves optimal performance across all evaluation metrics and get the fastest runtime. Furthermore, specific case studies demonstrate that, our method is able to predict more true positive and true negative sites and fewer false positives and false negatives sites than the current best method, proving that our model achieves better performance in PPIS prediction task.Currently, the data and core model code can be viewed here. The specific training code for this work is being sorted out and will be uploaded later.
### environment
Build the environment according to the configuration in the environment.yml.
### test
Run test.py to predict the model on the test set, with the test dataset in the fold Dataset.

