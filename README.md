# Bhagavad Gita AI Wisdom Guide 🌿📖🤖

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/vicky1603/bhagavad-gita-ai-wisdom-guide)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered assistant that provides personalized guidance from the Bhagavad Gita using modern NLP techniques. Part of Google's 5-Day GenAI Intensive Course (March-April 2025).

## Features ✨

- **Semantic Search**: Finds most relevant Gita verses for your life questions
- **Multilingual Support**: Sanskrit/Hindi/English outputs
- **Practical Wisdom**: Generates both philosophical insights and actionable advice
- **Structured Output**: Clean formatting with emoji markers (📜, 💡)

## Tech Stack 🛠️

| Component               | Technology Used                          |
|-------------------------|------------------------------------------|
| Embeddings              | `paraphrase-multilingual-mpnet-base-v2` |
| Vector Search           | Cosine Similarity                        |
| LLM                    | Gemini 1.5 Flash                         |
| GPU Acceleration        | PyTorch (CUDA)                           |
| Hosting                | Kaggle Notebook                          |

## Demo 🎥

**Sample Interaction:**

Ask your question: "How to deal with stress?"

📜 Bhagavad Gita 6.5: "Elevate yourself through the power of your mind..."
💡 Insight: The mind can be your greatest ally

Practice morning mindfulness (5 min)

Label emotions without judgment

Take short walk breaks hourly


## Installation ⚙️

1. Clone repository:
```bash
git clone https://github.com/Vicky16032205/Gen-AI-Intensive-Course-Capstone.git
cd Gen-AI-Intensive-Course-Capstone
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run python file
```bash
python main.py
```

4. You don't need to save your Gemini API key because when you will run the python file then it will itself ask for api key and then you have to enter it there only.

## Project Structure 📂
```
.Gen-AI-Intensive-Course-Capstone/
├── bhagavad-gita-ai-wisdom-guide.ipynb
├── data/
   └── all json files here
├── main.py
└── requirements.txt
```

## Capstone Requirements Checklist ✅

| Requirements               | Implementation                         |
|-------------------------|------------------------------------------|
| 3+ GenAI Capabilities   | RAG, Structured Output, Few-shot Prompting |
| Compilable Notebook     | Kaggle GPU-optimized notebook included   |
| Documentation	          | Detailed README + code comments          |

## Limitations & Future Work 🔮

- Currently text-only (future: voice input)
- Limited to 3 translations (future: more commentaries)
- Basic RAG (future: hybrid search)

## License 📜

MIT License - see LICENSE for details.

## Acknowledgments 🙏

- Google Cloud AI/ML team for course materials
- Kaggle for GPU resources
- Sanskrit scholars whose translations made this possible

## 🔗 Project Links

- 📺 [Watch the YouTube Demo](https://youtu.be/HYLJCqLbVZI)
- 📊 [Explore the Kaggle Notebook](https://www.kaggle.com/code/vicky1603/bhagavad-gita-ai-wisdom-guide)
- 📝 [Read the Blog Post](https://dev.to/vicky_gupta/unlocking-ancient-wisdom-with-ai-your-personal-bhagavad-gita-guide-for-modern-life-45k2)
