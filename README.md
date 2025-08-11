🤖 Simple Chatbot using NLP & Machine Learning

This project is a basic chatbot built using Python, Natural Language Processing (NLP), and Machine Learning. It understands user queries based on predefined intents and responds accordingly using a trained ML model.

---

## 📁 Project Structure

Task 3/
├── chatbot_model.h5 # Trained Keras model
├── chatbot.py # Main chatbot interaction script
├── train_chatbot.py # Model training script
├── intents.json # Chatbot intents (patterns, responses)
├── classes.pkl # Encoded class labels
├── words.pkl # Tokenized words list
├── requirements.txt # List of required Python packages
└── README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ How It Works

1. **Data**: `intents.json` contains predefined user intents with sample phrases and responses.
2. **Preprocessing**: Text data is tokenized and lemmatized using `nltk`.
3. **Model Training**: A simple neural network (via TensorFlow/Keras) is trained to classify the intent of the input.
4. **Inference**: When a user sends a message, the model predicts the intent and returns an appropriate response.

---

## 🧪 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
Required libraries:

nltk

numpy

tensorflow

🚀 How to Run
1. Train the chatbot (Only needed once or when updating intents):
bash
Copy
Edit
python train_chatbot.py
2. Start chatting with the bot:
bash
Copy
Edit
python chatbot.py
Type messages into the terminal. To exit, type:

bash
Copy
Edit
quit
🧠 Sample Intents
json
Copy
Edit
{
  "tag": "greeting",
  "patterns": ["Hi", "Hello", "Hey"],
  "responses": ["Hello!", "Hi there!", "Hey! How can I help you?"]
}
You can expand intents.json with more categories like:

Goodbye

Thanks

Name

Help

Age

Jokes

📌 Notes
Model is saved as chatbot_model.h5 after training.

Trained words and classes are saved using pickle for re-use.

You can improve the chatbot by adding more intents and better patterns.

👨‍💻 Author
Developed by PIYUSH CHAUDHARY as part of Task 3 – Internship Chatbot Project.