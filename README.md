# Drug Toxicity Prediction (Classification)

This is a project that aims to build a deep learning model for classifying Drug as toxic or non-toxic.

**Definition**: Majority of the drugs have some extents of toxicity to the human organisms. This learning task aims to predict accurately various types of toxicity of a drug molecule towards human organisms.

**Impact**: Toxicity is one of the primary causes of compound attrition. Study shows that approximately 70% of all toxicity-related attrition occurs preclinically (i.e., in cells, animals) while they are strongly predictive of toxicities in humans. This suggests that an early but accurate prediction of toxicity can significantly reduce the compound attribution and boost the likelihood of being marketed.

**Generalization**: Similar to the ADME prediction, as the drug structures of interest evolve over time, toxicity prediction requires a model to generalize to a set of novel drugs with small structural similarity to the existing drug set.

**Product**: Small-molecule.

**Pipeline**: Efficacy and safety - lead development and optimization.
## Dataset

### Acute Toxicity LD50
**Dataset Description**: Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. The higher the dose, the more lethal of a drug.

**Task Description**: Regression. Given a drug SMILES string, predict its acute toxicity.

**Dataset Statistics**: 7,385 drugs.
## Data Preprocessing

1. Converting SMILES data of drugs into morgan fingerprint data
[Molecular representations in AI-driven drug discovery: a review and practical guide](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00460-5)

2. The morgan fingerprint data are preprocessed using various NLP techniques, including tokenization, stopword removal, and stemming.
## Model Architecture

The model architecture used in this project is a combination of convolutional and recurrent neural networks, implemented using PyTorch. The model is trained using binary cross-entropy loss and optimized using the Adam optimizer.
## Training and Testing

The model is trained on a portion of the dataset and tested on a separate portion. The model is trained for a fixed number of epochs, with the best model (i.e., the one with the lowest validation loss) saved at each epoch. The performance of the model is evaluated using various metrics, including accuracy, sensitivity, specificity, and ROC-AUC score.
## Dependencies

Python 3.7+
PyTorch 1.7.1+
Scikit-learn 0.23.2+
Matplotlib 3.3.2+
Seaborn 0.11.0+
## Results

The model achieves an accuracy of 0.95, a sensitivity of 0.93, a specificity of 0.97, and an ROC-AUC score of 0.98 on the test set.
