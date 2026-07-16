# Reasoning Rhortcuts in Fashion-MNIST dataset

## 3-Digit Addition with FashionMNIST

The following experiments use the [Fashion-MNIST](https://arxiv.org/pdf/1708.07747) dataset to study Reasoning Shortcuts in the addition task: sum of three digits. 
All experiments use [Scallop](https://doi.org/10.1145/3591280), a language for NeuroSymbolic programming that integrates deep learning and reasoning.

### Experiment 1 pipeline:

**Shared Instance**: A single `MNISTNet` object is reused.  
Each digit image is passed separately, producing three predictions via **three forward calls**:

```python
self.mnist_net = MNISTNet()

distrs_a = self.mnist_net(imgs_a)
distrs_b = self.mnist_net(imgs_b)
distrs_c = self.mnist_net(imgs_c)
```

| Input |   | Network |   | Predicted Concept <br> Distribution |   |  Logic Program |   | Output |
|:-----:|:-:|:-------:|:-:|:-----------------:|:-:|:--------------:|:-:|:------:|
| img_a <br> img_b <br> img_c | ↘ <br> &rarr; <br> ↗ | `MNISTNet` <br> | ↗ <br> &rarr; <br> ↘ | distr_a <br> distr_b <br> distr_c | ↘ <br> &rarr; <br> ↗ | c1 + c2 + c3 = y | &rarr; | ŷ |

### Experiment 2 pipeline:

**Modified Network for 3 Digits**: `MNISTNet` adapted to process three images in one forward pass.

```python
self.mnist_net3 = MNISTNet()

distrs_a, distrs_b, distrs_c = self.mnist_net3(imgs_a, imgs_b, imgs_c)
```

| Input |   | Network |   | Predicted Concept |   |  Logic Program |   | Output |
|:-----:|:-:|:-------:|:-:|:-----------------:|:-:|:--------------:|:-:|:------:|
| img_a <br> img_b <br> img_c | ↘ <br> &rarr; <br> ↗ | `MNISTNet` <br> | ↗ <br> &rarr; <br> ↘ | distr_a <br> distr_b <br> distr_c | ↘ <br> &rarr; <br> ↗ | c1 + c2 + c3 = y | &rarr; | ŷ |

### Experiment 3 pipeline:

**Three Independent Instances**: Three separate `MNISTNet` networks (one per digit).

```python
self.mnist_net_d1 = MNISTNet()
self.mnist_net_d2 = MNISTNet()
self.mnist_net_d3 = MNISTNet()

distrs_a = self.mnist_net_d1(imgs_a)
distrs_b = self.mnist_net_d2(imgs_b)
distrs_c = self.mnist_net_d3(imgs_c)
```

| Input |   | Network |   | Predicted Concept |   |  Logic Program |   | Output |
|:-----:|:-:|:-------:|:-:|:-----------------:|:-:|:--------------:|:-:|:------:|
| img_a <br> img_b <br> img_c | &rarr; <br> &rarr; <br> &rarr; | `MNISTNetDigit1` <br> `MNISTNetDigit2` <br> `MNISTNetDigit3` | &rarr; <br> &rarr; <br> &rarr; | distr_a <br> distr_b <br> distr_c | &rarr; <br> &rarr; <br> &rarr; | c1 + c2 + c3 = y | &rarr; | ŷ |

### Experiment 4 pipeline:

**Three Independent Instances**: Three separate `MNISTNet` networks (one per digit).

```python
self.mnist_net_d1 = MNISTNet()
self.mnist_net_d2 = MNISTNet()
self.mnist_net_d3 = MNISTNet()

distrs_a = self.mnist_net_d1(x)
distrs_b = self.mnist_net_d2(x)
distrs_c = self.mnist_net_d3(x)
```

| Input |   | Network |   | Predicted Concept |   |  Logic Program |   | Output |
|:-----:|:-:|:-------:|:-:|:-----------------:|:-:|:--------------:|:-:|:------:|
| x = [img_a, img_b, img_c] | &rarr; <br> &rarr; <br> &rarr; | `MNISTNetDigit1` <br> `MNISTNetDigit2` <br> `MNISTNetDigit3` | &rarr; <br> &rarr; <br> &rarr; | distr_a <br> distr_b <br> distr_c | &rarr; <br> &rarr; <br> &rarr; | c1 + c2 + c3 = y | &rarr; | ŷ |

---

### Notes
- In experiments 1-3, the input consists of **three images**: `(img_a, img_b, img_c)`.
- In experiment 4, the input consists of a **concatenated image**: `tensor(img_a, img_b, img_c)`.
- The difference lies in how the recognition network’s weights are shared or duplicated.
- The final output connects to the logical module `sum_3`, producing a tensor of size **64 × 28** (possible sums from 0 to 27).

---

## Results

### Experiment 1:

### Experiment 2:
Total Training Time: 3074.61 s
Total Test Time: 144.73 s

### Experiment 3:

Total Training Time: 2282.47 s
Total Test Time: 97.47 s

Total Training Time: 2315.34 s
Total Test Time: 102.37 s

### Experiment 4:

Total Training Time: 3356.78 s
Total Test Time: 160.70 s

