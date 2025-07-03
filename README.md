# Content-Based Product Recommender System with PySpark & OpenAI

A scalable product recommender system built using Apache PySpark that leverages the power of OpenAI's text embeddings to understand product meaning and recommend items based on semantic similarity.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

## Overview

This project demonstrates how to build a modern, content-based recommender system. Traditional systems often rely on simple keyword matching. This system goes a step further by using advanced NLP to understand the *context* and *meaning* behind product titles and descriptions.

The core idea is to convert product text into meaningful numerical representations (embeddings) and then group similar products into clusters. When a user shows interest in a product, we can recommend other items from the same cluster, ensuring the recommendations are relevant and conceptually similar.

This project is built with scalability in mind, using PySpark to handle datasets that could be too large for a single machine.

## How It Works

The recommendation pipeline follows these key steps:

1.  **Data Loading & Preparation**: The initial product dataset (containing product IDs, titles, etc.) is loaded into a PySpark DataFrame for scalable processing.

2.  **Text Embedding with OpenAI**:
    * Product titles and descriptions are combined into a single text field.
    * This text is sent to an OpenAI embedding model (`text-embedding-3-small`).
    * The model returns a high-dimensional vector (512 dimensions) for each product. This vector numerically represents the semantic meaning of the product's text.

3.  **Dimensionality Reduction with PCA**:
    * Working with 512 dimensions is computationally expensive and can be noisy.
    * Principal Component Analysis (PCA) is used to reduce the dimensionality of the embedding vectors (down to 2 dimensions) while preserving the most important information.

4.  **Clustering with K-Means**:
    * The K-Means algorithm is applied to the reduced-dimension vectors.
    * This groups the products into a predefined number of clusters (`k`). Products within the same cluster are considered semantically similar to each other.

5.  **Generating Recommendations**:
    * The system identifies which cluster a user's recently viewed product belongs to.
    * It then recommends other products from that same cluster, providing relevant and context-aware suggestions.

## Technologies Used

* **Data Processing & ML:**
    * [Apache Spark (PySpark)](https://spark.apache.org/docs/latest/api/python/): For distributed data processing and building a scalable ML pipeline.
    * [Pandas](https://pandas.pydata.org/): For smaller, in-memory data manipulations and analysis.
    * [NumPy](https://numpy.org/): For numerical operations.
* **NLP & Embeddings:**
    * [OpenAI API](https://platform.openai.com/): For generating high-quality text embeddings.
* **Machine Learning:**
    * [Apache Spark (PySpark)](https://spark.apache.org/docs/latest/api/python/): For implementing PCA and K-Means clustering.
* **Data Visualization:**
    * [Plotly](https://plotly.com/python/): For creating interactive visualizations of the product clusters.

## Setup and Installation

Follow these steps to set up the project environment on your local machine.

#### 1. Prerequisites

* **Python 3.9+**
* **Java 11 JDK**: PySpark requires a Java installation to run.
    * Verify your installation with `java -version`.
    * Ensure the `JAVA_HOME` environment variable is set correctly.
* **Hadoop `winutils.exe` (for Windows users)**:
    * Download `winutils.exe` and place it in a folder (e.g., `C:\hadoop\bin`).
    * Set the `HADOOP_HOME` environment variable to point to this folder (e.g., `C:\hadoop`).

#### 2. Set Up a Virtual Environment and activate

It is highly recommended to use a virtual environment.

#### 3. Install Dependencies

Install all the required Python libraries.

#### 4. Set Up Environment Variables

You need to set your OpenAI API key. The best way to do this is with a `.env` file.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your API key to this file:
    ```
    OPENAI_API_KEY="your-secret-api-key-here"
    ```
The Python script will automatically load this key.

## How to Run

Once the setup is complete, you can run the project using the main Jupyter Notebook or Python script.

1.  **Launch Jupyter Notebook:**
    ```
    jupyter notebook
    ```
2.  Open the main notebook file (`recommender_system.ipynb`).
3.  Run the cells in order from top to bottom.

## Future Improvements

This project provides a solid foundation. Here are some ways it could be extended:

* **Build a User Interface:** Create a simple web application (using Flask or Streamlit) where a user can select a product and see the recommendations in real-time.
* **Implement Collaborative Filtering:** Combine this content-based approach with collaborative filtering (which uses user-item interaction data) to create a more powerful hybrid recommender system.
* **Use Open-Source Embedding Models:** To make the solution more cost-effective, replace the OpenAI API call with a self-hosted model from the `sentence-transformers` library (e.g., `all-MiniLM-L6-v2`).
* **Automate the Pipeline:** Convert the notebook into a script that can be run on a schedule (e.g., using Airflow) to regularly update the product clusters as new items are added.



