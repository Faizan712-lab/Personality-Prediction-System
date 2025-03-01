import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from docx import Document
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Sample dataset (Replace with a real, larger dataset for better accuracy)
data = [
    ("Team player with leadership qualities", "Extroversion"),
    ("Prefers working alone and enjoys deep analysis", "Introversion"),
    ("Strong attention to detail and organization", "Conscientiousness"),
    ("Creative thinker with an open mind", "Openness"),
    ("Highly cooperative and empathetic", "Agreeableness"),
    ("Struggles with stress and anxiety", "Neuroticism"),
]

# Data preprocessing
texts, labels = zip(*data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Model creation
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Prediction function with confidence score
def predict_personality(text):
    text = preprocess_text(text)
    prediction = model.predict([text])
    probabilities = model.predict_proba([text])[0]
    confidence = max(probabilities) * 100
    return prediction[0], confidence

# File selection and analysis
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Word Documents", "*.docx")])
    if file_path:
        try:
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            result, confidence = predict_personality(text)
            result_label.config(text=f"Predicted Personality: {result}\nConfidence: {confidence:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze the file: {e}")

# Exporting the result to a text file
def save_result():
    result_text = result_label.cget("text")
    if result_text and "Predicted Personality" in result_text:
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "w") as file:
                file.write(result_text)
            messagebox.showinfo("Success", "Result saved successfully!")
    else:
        messagebox.showwarning("Warning", "No prediction to save!")

# GUI setup
app = tk.Tk()
app.title("Personality Prediction System")
app.geometry("450x250")
app.configure(bg="#f2f2f2")

# Title
title_label = tk.Label(app, text="Personality Prediction via CV Analysis",
                       font=("Arial", 16, "bold"), bg="#f2f2f2", fg="#333")
title_label.pack(pady=10)

# Open File Button
open_button = tk.Button(app, text="Select CV (.docx)", command=open_file,
                        font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
open_button.pack(pady=10)

# Result Display
result_label = tk.Label(app, text="Predicted Personality: ",
                        font=("Arial", 12), bg="#f2f2f2", fg="#555")
result_label.pack(pady=10)

# Save Result Button
save_button = tk.Button(app, text="Save Result", command=save_result,
                        font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5)
save_button.pack(pady=10)

app.mainloop()
