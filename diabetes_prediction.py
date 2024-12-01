import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from experta import KnowledgeEngine, Rule, Fact, P  # Import P here
import tkinter as tk
from tkinter import messagebox, font

# Load Pima Indians Diabetes Dataset from a local file
file_path = "D:\Documents\semister 5\Reasoning skill\Diabetes Prediction Semester Project\Diabetes Prediction Semester Project\diabetes.csv"
data = pd.read_csv(file_path)

# Separate features (X) and target (y) from the dataset
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the test set results
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)

# Knowledge Base for Reasoning
class DiabetesKnowledgeEngine(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.reasons = []  # List to store reasoning explanations

    # Rule to check for high glucose level
    @Rule(Fact(glucose=P(lambda x: x > 140)))
    def high_glucose(self):
        self.reasons.append("High glucose level detected, risk of diabetes.")

    # Rule to check for high BMI
    @Rule(Fact(bmi=P(lambda x: x > 30)))
    def high_bmi(self):
        self.reasons.append("BMI indicates overweight, increasing diabetes risk.")

    # Rule to check for high age
    @Rule(Fact(age=P(lambda x: x > 45)))
    def high_age(self):
        self.reasons.append("Age is a risk factor for diabetes.")

    # Rule for family history of diabetes with clarification
    @Rule(Fact(family_history=True), Fact(glucose=P(lambda x: x <= 140)), Fact(bmi=P(lambda x: x <= 30)), Fact(age=P(lambda x: x <= 45)))
    def family_history_with_no_other_risks(self):
        self.reasons.append("Family history of diabetes detected. While this increases likelihood, no other risk factors were identified. Consider monitoring and regular check-ups.")

    # Rule for borderline age and BMI
    @Rule(Fact(age=P(lambda x: x == 45)))
    def borderline_age(self):
        self.reasons.append("Age is 45, which is borderline. May require additional monitoring for diabetes risk.")

    @Rule(Fact(bmi=P(lambda x: x == 30)))
    def borderline_bmi(self):
        self.reasons.append("BMI is 30, which is borderline. Lifestyle changes like regular exercise and a balanced diet can help reduce diabetes risk.")

    # Rule for no significant risk factors
    @Rule(Fact(glucose=P(lambda x: x <= 140)), Fact(bmi=P(lambda x: x <= 30)), Fact(age=P(lambda x: x <= 45)))
    def no_significant_risk(self):
        self.reasons.append("No significant risks identified. Keep maintaining a healthy lifestyle to minimize the chances of diabetes.")

    # Provide preventive measures
    @Rule(Fact(glucose=P(lambda x: x > 140)))
    def provide_preventive_measures(self):
        self.reasons.append("It is recommended to monitor your blood sugar levels, maintain a balanced diet, and consult a doctor for personalized advice.")

    @Rule(Fact(bmi=P(lambda x: x > 30)))
    def provide_preventive_measures_bmi(self):
        self.reasons.append("Maintaining a healthy BMI is important. Consider regular physical activity and consult a healthcare provider for a tailored fitness plan.")

# GUI for User Input and Prediction
class HealthGuardApp:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("HealthGuard: Diabetes Prediction")
        self.root.geometry("600x700")
        self.root.configure(bg='#2e2e2e')

        # Define fonts and colors
        self.header_font = ("Segoe UI", 18, "bold", "underline")
        self.subheader_font = ("Segoe UI", 14, "bold")
        self.normal_font = ("Segoe UI", 12)
        self.button_font = ("Segoe UI", 12, "bold")
        self.bg_color = "#2e2e2e"
        self.text_color = "#ffffff"
        self.button_color = "#0052cc"
        self.entry_bg_color = "#404040"
        self.entry_text_color = "#ffffff"

        self.create_widgets()

    def create_widgets(self):
        # Header label
        tk.Label(self.root, text="Diabetes Prediction", font=self.header_font, bg=self.bg_color, fg=self.text_color).pack(pady=20)

        # Frame for inputs
        form_frame = tk.Frame(self.root, bg=self.bg_color)
        form_frame.pack(pady=10)

        # Dictionary for input labels and entries
        self.labels_entries = {}
        self.columns = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        questions = [
            "Number of Pregnancies:", "Plasma Glucose Concentration:", "Diastolic Blood Pressure (mm Hg):",
            "Triceps Skinfold Thickness (mm):", "2-hour Serum Insulin (mu U/ml):",
            "Body Mass Index (BMI):", "Diabetes Pedigree Function:", "Age:"
        ]

        # Create input fields
        for question, column in zip(questions, self.columns):
            frame = tk.Frame(form_frame, bg=self.bg_color)
            frame.pack(pady=5, fill="x")
            label = tk.Label(frame, text=question, font=self.normal_font, bg=self.bg_color, fg=self.text_color)
            label.pack(side=tk.LEFT, padx=10)
            entry = tk.Entry(frame, font=self.normal_font, bg=self.entry_bg_color, fg=self.entry_text_color)
            entry.pack(side=tk.RIGHT, padx=10, fill="x", expand=True)
            self.labels_entries[column] = entry

        # Prediction button
        tk.Button(self.root, text="Predict", font=self.button_font, bg=self.button_color, fg=self.text_color, 
                  command=self.predict_disease).pack(pady=20)

        # Prediction result
        self.result_label = tk.Label(self.root, text="", font=self.subheader_font, bg=self.bg_color, fg=self.text_color)
        self.result_label.pack(pady=10)

        # Reasoning output
        self.reasoning_label = tk.Label(self.root, text="Reasoning Output:", font=self.subheader_font, bg=self.bg_color, fg=self.text_color)
        self.reasoning_label.pack(pady=10)
        self.reasoning_text = tk.Text(self.root, height=10, font=self.normal_font, bg=self.entry_bg_color, fg=self.text_color, wrap="word")
        self.reasoning_text.pack(pady=10, fill="x", padx=20)

    def predict_disease(self):
        try:
            # Collect user input
            features = [float(self.labels_entries[col].get()) for col in self.columns]
            input_data = pd.DataFrame([features], columns=self.columns)
            input_data = scaler.transform(input_data)

            # Model prediction
            prediction = model.predict(input_data)[0]
            result_text = f"Predicted Disease: {'Positive' if prediction == 1 else 'Negative'}"
            self.result_label.config(text=result_text)

            # Reasoning engine
            engine = DiabetesKnowledgeEngine()
            engine.reset()
            engine.declare(Fact(glucose=features[1]), Fact(bmi=features[5]), Fact(age=features[7]), Fact(family_history=True))
            engine.run()

            # Display reasoning
            reasoning_output = "\n".join(engine.reasons) if engine.reasons else "No significant risk factors detected."
            self.reasoning_text.delete("1.0", tk.END)
            self.reasoning_text.insert(tk.END, reasoning_output)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric data.")

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = HealthGuardApp(root)
    root.mainloop()