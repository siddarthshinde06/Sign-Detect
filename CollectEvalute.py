import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import threading
import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from collections import deque

# Global variables
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection UI (Tkinter Fallback)")
        self.root.geometry("700x600")
        self.root.configure(bg="#f0f0f0")

        # Style for better look
        style = ttk.Style()
        style.configure("TNotebook", background="#f0f0f0")
        style.configure("TButton", font=("Arial", 10), padding=5)
        style.configure("TLabel", font=("Arial", 12), background="#f0f0f0")

        # Notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tabs
        self.data_tab = ttk.Frame(self.notebook, style="TFrame")
        self.train_tab = ttk.Frame(self.notebook, style="TFrame")
        self.predict_tab = ttk.Frame(self.notebook, style="TFrame")
        self.settings_tab = ttk.Frame(self.notebook, style="TFrame")

        self.notebook.add(self.data_tab, text="Data Collection")
        self.notebook.add(self.train_tab, text="Model Training")
        self.notebook.add(self.predict_tab, text="Real-Time Prediction")
        self.notebook.add(self.settings_tab, text="Settings & Stats")

        # Setup each tab
        self.setup_data_collection_tab()
        self.setup_training_tab()
        self.setup_prediction_tab()
        self.setup_settings_tab()

    def setup_data_collection_tab(self):
        tab = self.data_tab
        ttk.Label(tab, text="Collect Hand Gesture Data", font=("Arial", 16, "bold")).pack(pady=10)

        # Label for instruction
        ttk.Label(tab, text="Enter gesture label (e.g., A, Hello):", font=("Arial", 10)).pack(pady=5)
        self.label_entry = ttk.Entry(tab)
        self.label_entry.pack(pady=5)

        # Label for samples
        ttk.Label(tab, text="Target samples (default: 100):", font=("Arial", 10)).pack(pady=5)
        self.samples_entry = ttk.Entry(tab)
        self.samples_entry.insert(0, "100")
        self.samples_entry.pack(pady=5)

        self.progress_bar = ttk.Progressbar(tab, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=10)

        ttk.Button(tab, text="Start Collection", command=self.collect_data).pack(pady=10)

        self.collection_status = tk.Text(tab, height=10, wrap="word", bg="#ffffff")
        self.collection_status.pack(pady=10, fill="both", expand=True)

    def setup_training_tab(self):
        tab = self.train_tab
        ttk.Label(tab, text="Train SVM Model", font=("Arial", 16, "bold")).pack(pady=10)

        self.train_progress = ttk.Progressbar(tab, orient="horizontal", length=400, mode="determinate")
        self.train_progress.pack(pady=10)

        ttk.Button(tab, text="Train Model", command=self.train_model).pack(pady=10)

        self.train_status = tk.Text(tab, height=15, wrap="word", bg="#ffffff")
        self.train_status.pack(pady=10, fill="both", expand=True)

    def setup_prediction_tab(self):
        tab = self.predict_tab
        ttk.Label(tab, text="Live Sign Detection", font=("Arial", 16, "bold")).pack(pady=10)

        ttk.Button(tab, text="Start Prediction", command=self.realtime_predict).pack(pady=10)

        self.predict_status = tk.Text(tab, height=15, wrap="word", bg="#ffffff")
        self.predict_status.pack(pady=10, fill="both", expand=True)

    def setup_settings_tab(self):
        tab = self.settings_tab
        ttk.Label(tab, text="App Settings and Dataset Stats", font=("Arial", 16, "bold")).pack(pady=10)

        ttk.Button(tab, text="View Dataset Stats", command=self.view_stats).pack(pady=10)
        ttk.Button(tab, text="Clear Dataset", command=self.clear_dataset).pack(pady=5)
        ttk.Button(tab, text="Export Results", command=self.export_results).pack(pady=5)

        self.stats_text = tk.Text(tab, height=15, wrap="word", bg="#ffffff")
        self.stats_text.pack(pady=10, fill="both", expand=True)

    def update_status(self, textbox, message):
        textbox.insert(tk.END, message + "\n")
        textbox.see(tk.END)

    def collect_data(self):
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showerror("Error", "Please enter a valid label.")
            return
        try:
            target_samples = int(self.samples_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of samples.")
            return
        if messagebox.askyesno("Confirm", f"Start collecting {target_samples} samples for '{label}'?"):
            threading.Thread(target=self._collect_data_thread, args=(label, target_samples)).start()

    def _collect_data_thread(self, label, target_samples):
        self.update_status(self.collection_status, f"Starting data collection for '{label}'...")
        self.progress_bar['value'] = 0

        if not os.path.exists("dataset.csv"):
            with open("dataset.csv", "w", newline="") as f:
                writer = csv.writer(f)
                header = [f"landmark_{i}" for i in range(63)] + ["label"]
                writer.writerow(header)

        cap = cv2.VideoCapture(0)
        samples_collected = 0

        while samples_collected < target_samples:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    wrist = np.array([hand.landmark[0].x, hand.landmark[0].y, hand.landmark[0].z])
                    normalized_data = []
                    for lm in hand.landmark:
                        normalized_data.extend((np.array([lm.x, lm.y, lm.z]) - wrist).tolist())

                    scale_factor = np.linalg.norm(np.array([hand.landmark[9].x, hand.landmark[9].y, hand.landmark[9].z]) - wrist)
                    if scale_factor > 0:
                        normalized_data = [val / scale_factor for val in normalized_data]

                    normalized_data.append(label)

                    with open("dataset.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(normalized_data)
                        flipped_data = [-normalized_data[i] if i % 3 == 0 else normalized_data[i] for i in range(len(normalized_data)-1)] + [label]
                        writer.writerow(flipped_data)
                        noisy_data = [val + np.random.normal(0, 0.01) for val in normalized_data[:-1]] + [label]
                        writer.writerow(noisy_data)

                    samples_collected += 3
                    self.progress_bar['value'] = (samples_collected / target_samples) * 100
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Samples: {samples_collected}/{target_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Collecting Data", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.update_status(self.collection_status, f"Data collection complete for '{label}'. Total samples: {samples_collected}")
        self.progress_bar['value'] = 100

    def train_model(self):
        if not os.path.exists("dataset.csv"):
            messagebox.showerror("Error", "No dataset found. Collect data first.")
            return
        if messagebox.askyesno("Confirm", "Start training? This may take time."):
            threading.Thread(target=self._train_model_thread).start()

    def _train_model_thread(self):
        self.update_status(self.train_status, "Training model...")
        self.train_progress['value'] = 20

        df = pd.read_csv("dataset.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.train_progress['value'] = 40

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
        self.train_progress['value'] = 60

        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
        grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
        grid.fit(X_train, y_train)
        self.train_progress['value'] = 80

        model = grid.best_estimator_
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        joblib.dump(model, "svm_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        self.train_progress['value'] = 100

        self.update_status(self.train_status, f"Training complete. Best params: {grid.best_params_}. Accuracy: {acc:.4f}\n{confusion_matrix(y_test, y_pred)}")

    def realtime_predict(self):
        if not os.path.exists("svm_model.pkl") or not os.path.exists("scaler.pkl"):
            messagebox.showerror("Error", "Model not trained. Train the model first.")
            return
        threading.Thread(target=self._realtime_predict_thread).start()

    def _realtime_predict_thread(self):
        self.update_status(self.predict_status, "Starting real-time prediction...")

        model = joblib.load("svm_model.pkl")
        scaler = joblib.load("scaler.pkl")

        cap = cv2.VideoCapture(0)
        prediction_buffer = deque(maxlen=5)
        confidence_threshold = 0.8

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    wrist = np.array([hand.landmark[0].x, hand.landmark[0].y, hand.landmark[0].z])
                    normalized_data = []
                    for lm in hand.landmark:
                        normalized_data.extend((np.array([lm.x, lm.y, lm.z]) - wrist).tolist())

                    scale_factor = np.linalg.norm(np.array([hand.landmark[9].x, hand.landmark[9].y, hand.landmark[9].z]) - wrist)
                    if scale_factor > 0:
                        normalized_data = [val / scale_factor for val in normalized_data]

                    normalized_data = scaler.transform([normalized_data])

                    if len(normalized_data[0]) == 63:
                        pred = model.predict(normalized_data)[0]
                        prob = max(model.predict_proba(normalized_data)[0])

                        prediction_buffer.append((pred, prob))

                        if len(prediction_buffer) == 5:
                            avg_pred = max(set([p[0] for p in prediction_buffer]), key=[p[0] for p in prediction_buffer].count)
                            avg_prob = np.mean([p[1] for p in prediction_buffer])

                            if avg_prob >= confidence_threshold:
                                cv2.putText(frame, f"{avg_pred} ({avg_prob:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, "Low Confidence", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Real-Time Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.update_status(self.predict_status, "Real-time prediction stopped.")

    def view_stats(self):
        if not os.path.exists("dataset.csv"):
            self.update_status(self.stats_text, "No dataset found.")
            return
        df = pd.read_csv("dataset.csv")
        stats = f"Total samples: {len(df)}\nClass distribution:\n{df['label'].value_counts().to_string()}"
        self.update_status(self.stats_text, stats)

    def clear_dataset(self):
        if messagebox.askyesno("Confirm", "Delete dataset.csv? This cannot be undone."):
            if os.path.exists("dataset.csv"):
                os.remove("dataset.csv")
                self.update_status(self.stats_text, "Dataset cleared.")
            else:
                messagebox.showinfo("Info", "No dataset to clear.")

    def export_results(self):
        if not os.path.exists("svm_model.pkl"):
            messagebox.showerror("Error", "No trained model to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "w") as f:
                f.write("Model exported. Accuracy details in training tab.\n")
            messagebox.showinfo("Success", "Results exported.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
