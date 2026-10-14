# Hi, I'm Peixin Li 👋

M.S. student in Electronic Information at **Southwest University**.

My research interests focus on:

- Recommendation Systems
- Parallel Machine Learning
- Tensor Decomposition
- Large-scale Optimization

I mainly work with **Python, C++, PyTorch, CUDA and OpenMP**.

## 🔬 Research Interests

- Parallel SGD for large-scale recommendation systems
- Matrix Factorization
- Tensor Ring Decomposition
- Tensor Completion
- GPU-accelerated machine learning
- Large-scale recommendation experiments

## 🚀 Featured Projects

# Parallel SGD for Matrix Factorization

C++/OpenMP implementations and research artifacts for parallel stochastic
gradient descent in large-scale matrix factorization and recommendation systems.

This repository is associated with my first-author IEEE conference paper:

> **An Efficient Hybrid Parallel SGD Algorithm for Matrix Factorization in Recommendation Systems**  
> Peixin Li, Qicong Hu  
> 2025 6th International Symposium on Computer Engineering and Intelligent Communications (ISCEIC)  
> IEEE, 2025  
> DOI: `10.1109/ISCEIC67854.2025.11405719`

The project focuses on improving the training efficiency of matrix factorization
on large-scale sparse recommendation data using shared-memory parallelism.

The repository currently provides a cleaned and reproducible public portfolio
implementation of several SGD baselines together with experiment infrastructure.
The complete research implementation of A-ASGD is not included in this public
portfolio release.

---

## Research Paper

### An Efficient Hybrid Parallel SGD Algorithm for Matrix Factorization in Recommendation Systems

**Peixin Li**, Qicong Hu  
2025 6th International Symposium on Computer Engineering and Intelligent Communications (ISCEIC)  
IEEE, 2025  
DOI: `10.1109/ISCEIC67854.2025.11405719`

### Motivation

Matrix Factorization (MF) is widely used in recommendation systems to model
latent interactions between users and items.

However, conventional SGD becomes increasingly expensive on large-scale sparse
datasets such as MovieLens-20M.

Existing parallel SGD strategies face a trade-off:

- Fully lock-free methods such as **Hogwild!** provide high throughput but may
  suffer from conflicting parameter updates.
- Strongly synchronized approaches improve update consistency but introduce
  synchronization and coordination overhead.

The paper proposes **A-ASGD (Asymmetric Asynchronous SGD)** to balance these two
objectives.

### Key Idea

A-ASGD exploits the asymmetric sparsity structure of user-item interaction data.

Although user and item latent factors use mathematically symmetric SGD update
rules, their access patterns can be very different in real recommendation data.

In particular:

- **User-side updates are performed lock-free**, allowing maximum parallelism.
- **Item-side updates use fine-grained synchronization**, reducing conflicting
  updates on frequently accessed item factors.
- The user-item interaction space is partitioned into a **2N × 4N block grid**,
  where `N` is the number of worker threads.
- Workers randomly sample blocks during training.
- Item blocks use a **non-blocking try-lock strategy**. If a block is already
  locked, the worker immediately samples another block instead of waiting.

This produces an asymmetric parallel update strategy:

```text
                     User-Item Interaction Matrix
                                |
                                v
                     2N × 4N Grid Partition
                                |
                                v
                    Randomized Block Scheduler
                                |
                +---------------+---------------+
                |                               |
                v                               v
        User-side Parameters            Item-side Parameters
          M_u and b_u                     N_i and b_i
                |                               |
                v                               v
        Lock-free Updates            Fine-grained Block Lock
                |                               |
                +---------------+---------------+
                                |
                                v
                       Parallel SGD Training

### Tensor Ring Recommendation

PyTorch implementation of Tensor Ring decomposition for large-scale
recommendation and tensor completion.

**Topics:** PyTorch · CUDA · Tensor Ring · Tensor Completion · Recommendation Systems

> Project repository coming soon.

### Recommendation System Benchmarks

Experimental comparison of recommendation models on MovieLens and
Amazon datasets.

**Topics:** PyTorch · Matrix Factorization · RMSE · GPU Training

> Project repository coming soon.

## 🛠 Tech Stack

**Languages**

Python · C++ · SQL

**Machine Learning**

PyTorch · NumPy · Pandas · Scikit-learn

**Parallel Computing**

OpenMP · CUDA · Multi-threading

**Research**

Recommendation Systems · Matrix Factorization · Tensor Decomposition · Parallel SGD
