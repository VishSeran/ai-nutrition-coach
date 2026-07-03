# AI Nutrition Coach 🍎🤖

An AI-powered Nutrition Coach web application that estimates calories and provides dietary insights from food images using state-of-the-art Vision-Language Models from Hugging Face.

## 🚀 Project Overview

This project is a Flask-based web application that leverages generative AI and computer vision to analyze food images, identify food items, estimate caloric content, and provide personalized nutritional guidance.

Instead of IBM Watsonx AI, this implementation uses **Hugging Face Transformers** with vision-language models (e.g., LLaVA / Llama-family VLMs / CLIP-based pipelines) to perform food recognition and description tasks.

The recognized food items are mapped against a nutritional database to estimate calories and macronutrients.

## 🎯 Key Features

- 📷 Upload food images via a simple web interface
- 🧠 AI-powered food recognition using Hugging Face models
- 🔥 Calorie estimation based on nutritional databases
- 🥗 Macro breakdown (protein, carbs, fats)
- 💡 Personalized dietary suggestions
- 🌐 Flask-based lightweight web application

## 🏗️ Tech Stack

- Python 3.10+
- FastAPI
- Hugging Face Transformers
- PyTorch
- Pillow (PIL)
- Pandas / JSON nutrition dataset
- React/CSS (frontend)

## 🧠 AI Pipeline

1. User uploads food image
2. Image is processed using a Vision-Language Model (Hugging Face)
3. Model identifies food items in the image
4. Identified items are matched with a nutrition database
5. Calories and macros are computed
6. Results + dietary advice returned to user

## 📁 Project Structure

```
ai-nutrition-coach/
│── app.py
│── model/
│   ├── vision_model.py
│   ├── nutrition_mapper.py
│── data/
│   ├── nutrition_db.json
│── static/
│── templates/
│   ├── index.html
│── utils/
│   ├── image_utils.py
│── requirements.txt
│── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/your-username/ai-nutrition-coach.git
cd ai-nutrition-coach
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
python app.py
```

Then open:
```
http://127.0.0.1:5000
```

## 🧪 Example Workflow

- Upload: burger.jpg
- Output:
  - Food: Burger, Fries
  - Calories: ~850 kcal
  - Protein: 32g
  - Advice: High calorie meal, consider balancing with vegetables

## 📌 Future Improvements

- Add meal tracking history
- Integrate barcode scanning
- Improve accuracy with fine-tuned food models
- Deploy using Docker / cloud (AWS, GCP)

## ⚠️ Disclaimer

Calorie estimation is AI-based and may not be fully accurate. Always verify with professional nutritional tools for medical or dietary decisions.

---

Built with ❤️ using Hugging Face AI models
