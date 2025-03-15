# **Fine-Tuning LLM for Financial Text Classification**  

## **Overview**  
This project focuses on **fine-tuning a Large Language Model (LLM)** for **financial text classification**, leveraging **transfer learning techniques** to enhance domain-specific performance. Fine-tuning an LLM allows adapting a pre-trained model to a specialized dataset, improving its ability to understand and classify financial text effectively.  

The project follows the methodology outlined in this [Medium post](https://medium.com/data-science/fine-tuning-large-language-models-llms-23473d763b91), implementing **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter Efficient Fine-Tuning)** techniques to optimize model training while maintaining computational efficiency.

---

## **What is Fine-Tuning an LLM?**  
Fine-tuning an LLM involves taking a **pre-trained transformer-based language model** (such as GPT, BERT, or Mistral) and training it further on a **domain-specific dataset** to improve its understanding and response accuracy in that context.  

### **Why Fine-Tune?**  
- **Domain Adaptation**: General-purpose LLMs lack specialized financial knowledge. Fine-tuning aligns them with financial terminologies, risks, and compliance-related text.  
- **Improved Classification**: Fine-tuning enhances model accuracy for financial text classification tasks.  
- **Reduced Compute Costs**: Using efficient fine-tuning techniques like LoRA minimizes computational requirements.  

---

## **Fine-Tuning Method Used in This Project**  

### **1. Dataset Preparation**  
- The dataset consists of **financial documents, reports, and stock-related texts**, labeled into categories such as risk analysis, earnings reports, and market sentiment.  
- The text is **tokenized** using a tokenizer compatible with the chosen pre-trained model (e.g., `AutoTokenizer` from Hugging Face).  

### **2. LoRA (Low-Rank Adaptation) Fine-Tuning**  
- LoRA is a **Parameter Efficient Fine-Tuning (PEFT)** technique that modifies only **a small subset of model weights** while keeping the rest frozen.  
- It **injects small trainable rank-decomposition matrices** into the transformer layers, reducing **VRAM usage** and making the fine-tuning process more **efficient**.  

**Why LoRA?**  
✔ **Reduces trainable parameters** → Faster training, lower memory footprint.  
✔ **Maintains pre-trained model knowledge** while adapting to financial-specific text.  
✔ **Works well on low-resource GPUs** compared to full fine-tuning.  

### **3. PEFT (Parameter Efficient Fine-Tuning)**  
- PEFT strategies like **LoRA, Prefix Tuning, and Prompt Tuning** allow fine-tuning **without modifying the entire model**.  
- In this project, **LoRA is applied to specific transformer layers**, limiting memory consumption while **preserving general language knowledge** from pre-training.  

### **4. Training Process**  
- The **Hugging Face `transformers` library** is used to load the pre-trained model.  
- LoRA adapters are applied to **specific attention layers**.  
- The model is fine-tuned using **cross-entropy loss**, with hyperparameters optimized for best classification accuracy.  
- The **Trainer API from Hugging Face** manages training, evaluation, and gradient accumulation.  

---

## **Model Deployment & Evaluation**  
- The fine-tuned model is **evaluated** using standard NLP metrics:  
  - **Accuracy**  
  - **F1-score**  
  - **Precision/Recall**  
- The model can be deployed using **Hugging Face `pipeline` API** for real-time inference on financial text.  

---

## **Technologies Used**  
- **Hugging Face Transformers**  
- **LoRA & PEFT**  
- **PyTorch**  
- **Financial NLP Dataset**  
- **Google Colab / Local GPU**  

---

## **References**  
- [Fine-Tuning LLMs – Medium](https://medium.com/data-science/fine-tuning-large-language-models-llms-23473d763b91)  
- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)  
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)  
