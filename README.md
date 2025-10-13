# FinSecureAI: Your Personal AI Financial Analyst

FinSecureAI is a full-stack web application that leverages a fine-tuned Large Language Model to make complex financial analysis accessible to everyone. Ask questions in natural language about S&P 500 companies and receive intelligent, data-backed insights from official financial documents and market data.

**This project was developed as an extension of my Bachelor's Thesis, which scored a 9.85/10.**

### 🎥 Video Demo

[![FinSecureAI Demo Video](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID)

###  Key Features

*   **Conversational AI Chat:** Ask complex questions like *"What were Apple's main revenue drivers in Q4 2022?"* or *"Compare Tesla's and Ford's outlook on autonomous driving."*
*   **Retrieval-Augmented Generation (RAG):** The AI doesn't hallucinate. It grounds its answers in data retrieved from a comprehensive knowledge base of 10-K, 10-Q, and earnings call transcripts.
*   **Custom Fine-Tuned Model:** Utilizes a Llama-8B model fine-tuned with LoRA on a custom financial dataset for superior domain-specific understanding.
*   **Real-Time Market Data:** Integrates with yfinance to provide up-to-date stock prices, market cap, and other key metrics.

---

### 🏛️ System Architecture

The application is built on a modern, decoupled architecture ensuring scalability and maintainability.

![System Architecture Diagram](docs/architecture.png) 

---

### 🛠️ Tech Stack

| Area      | Technologies                                                                                           |
| :-------- | :----------------------------------------------------------------------------------------------------- |
| **Frontend** | ![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white) ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB) ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white) |
| **Backend**  | ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **AI/ML**    | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-FFD21E?style=for-the-badge) **(LoRA, 4-bit Quantization)** |
| **Database** | ![Qdrant](https://img.shields.io/badge/Qdrant-AC1431?style=for-the-badge) (Vector DB)  |
| **DevOps**   | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) ![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white) ![ngrok](https://img.shields.io/badge/ngrok-1F1E37?style=for-the-badge&logo=ngrok&logoColor=white) |

---
