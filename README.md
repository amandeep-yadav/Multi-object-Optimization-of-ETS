# Transformer-Based Multi-Objective Extractive Text Summarization

This project implements an automated Extractive Text Summarization (ETS) system using **Transformers** and **Non-Dominated Sorting Genetic Algorithm II (NSGA-II)**. The goal is to generate optimal summaries by maximizing **content coverage** while minimizing **redundancy**. The system combines the semantic power of transformers with multi-objective optimization to achieve state-of-the-art performance on text summarization tasks.

---

## 🚀 Features

- **Transformer-Based Similarity Matrix:** Leverages transformer embeddings (e.g., Sentence-BERT) to compute semantic similarity between sentences.
- **Multi-Objective Optimization:** Applies NSGA-II to balance coverage and redundancy.
- **Custom Termination Criteria:** Uses entropy-based dissimilarity for efficient convergence.
- **High Performance:** Achieves superior ROUGE scores compared to traditional methods.

---

## 📋 Requirements

Ensure you have the following installed:

- **Python** >= 3.8
- **PyTorch**
- **Transformers Library** (Hugging Face)
- **NumPy**
- **Matplotlib**
- **SciPy**

### Install Dependencies

Run the following command to install the necessary packages:

```bash
pip install torch transformers numpy matplotlib scipy
```

---

## 🛠️ Methodology

### 1. Text Preprocessing

- Tokenize input text into sentences.
- Remove stop words and punctuation.
- Apply stemming to reduce words to their root forms.

### 2. Sentence Representation

- **Sentence Embeddings:** Use Sentence-BERT to generate contextual sentence embeddings.
- **Similarity Matrix:** Compute pairwise cosine similarity between sentences.

### 3. Multi-Objective Optimization with NSGA-II

- **Objective 1:** Maximize content coverage by selecting sentences that capture the most relevant information.
- **Objective 2:** Minimize redundancy by avoiding repetitive information.
- Generate the Pareto front of optimal solutions.
- Employ entropy-based termination to ensure convergence.

---

## 📊 Results

The system was evaluated on the **DUC-2004 dataset** using ROUGE metrics:

| Metric      | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------------|----------|----------|----------|
| **Maximum** | 0.745    | 0.566    | 0.464    |
| **Average** | 0.663    | 0.464    | 0.401    |

### Pareto Front Visualization

The Pareto front demonstrates the trade-offs between coverage and redundancy across multiple generations:

![Test1](https://github.com/amandeep-yadav/Multi-object-Optimization-of-ETS/blob/main/img/test1.PNG)
![Test2](https://github.com/amandeep-yadav/Multi-object-Optimization-of-ETS/blob/main/img/test2.PNG)

---

## 📂 Project Structure

```plaintext
Transformer-ETS/
├── data/               # Input text datasets
├── models/             # Pretrained and fine-tuned transformer models
├── notebooks/          # Jupyter notebooks for experimentation
├── utils/              # Utility scripts for preprocessing and evaluation
├── results/            # Generated summaries and evaluation results
├── requirements.txt    # Dependency list
└── README.md           # Project documentation
```

---

## 📜 How to Use

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/Transformer-ETS.git
   cd Transformer-ETS
   ```

2. **Run the Notebook:**

   Open the Jupyter notebook for step-by-step guidance:

   ```bash
   jupyter notebook summarization_pipeline.ipynb
   ```

3. **Input Data:**

   Place your text data in the `data/` directory. The pipeline will tokenize, process, and generate summaries.

4. **Evaluate Summaries:**

   Compare generated summaries against reference summaries using ROUGE metrics.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## 📜 License

This project is licensed under the **MIT License**. See the LICENSE file for details.

---

## 💡 Acknowledgments

- **Hugging Face Transformers:** For providing state-of-the-art transformer models.
- **NSGA-II Framework:** For enabling multi-objective optimization.

---

## 📬 Contact

For questions, feedback, or collaboration, please reach out to: [Amandeep Yadav](mailto:your-email@example.com).

