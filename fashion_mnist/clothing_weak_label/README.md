# Experiment 

**Date**: 30/07/2026

**Name of the experiment**: Outfit valid

**Responsible**: Christian Barreto

**Aim**: Predict if outfit is valid through of three imagens used openCLIP

## Tool and configuration
**Languaje**: Python - 3.9

**Nesy**: Scallopy - 0.1.4

**Dataset**: MNIST-FASHION [reference][1].
![MNIST-FASHION](result/digit_graph.png)
**Neural model**:
 ```
OpenCLIP
model_name = "ViT-B-32"
pretrained = "laion2b_s34b_b79k"
num_classes = 10
freeze_backbone = False
 ```
**Rules logical**:
```
upper = {0,2,4,6}
lower = {1}
shoe  = {5,7,9}

digit_1(X),  X ∈ {0,1,2,3,4,5,6,7,8,9}
digit_2(Y),  Y ∈ {0,1,2,3,4,5,6,7,8,9}
digit_3(Z),  Z ∈ {0,1,2,3,4,5,6,7,8,9}

digit_1(X) ∧ upper(X) → has_upper(X)
digit_2(X) ∧ lower(X) → has_lower(X)
digit_3(X) ∧ shoe(X)  → has_shoe(X)

has_upper(U) ∧ has_lower(L) ∧ has_shoe(S) → valid
```
**Hyper parameters**:
- **Epoch**: 20
- **Bash size**: 64
- **Learning rate**: 0,0001
- **Loss function**: Binary Cross Entropy
- **Seed**: 1234
- **Provinence**: difftopkproofs
- **Top-k**: 3
## Method
### 1. Weak Label

### 2. Weak Beal



## Result
**Training**
![MNIST-FASHION](result/result_graph_train.png)

**Testing**

![MNIST-FASHION](result/result_graph_test.png)

**Confusion Matrix**

![MNIST-FASHION](result/result_matrix_test.png)

## Conclusions and notes

## References

[1]: https://arxiv.org/pdf/1708.07747 "MNIST-FASHION Dataset"
