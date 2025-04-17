# Arabic Handwriting Recognition System

![Project Banner](https://via.placeholder.com/1200x400/2D3748/FFFFFF?text=Arabic+Handwriting+Recognition+with+CNN-LSTM+and+Attention+Mechanism)

**Final year project** - A deep learning system for recognising handwritten Arabic characters using a hybrid CNN-LSTM architecture with optional attention mechanism, deployed as a Streamlit web application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arabic-handwriting-recognition.streamlit.app/) ← Live Demo


## 🚀 Features
| Feature | Description |
|---------|-------------|
| **Hybrid Architecture** | CNN for spatial features + LSTM for sequential patterns |
| **Attention Mechanism** | Optional attention layer for improved accuracy |
| **End-to-End Pipeline** | Data loading → Training → Testing → Prediction |
| **Interactive UI** | Drawing canvas + file upload + real-time visualization | 

  - **Hybrid Architecture**: Combines CNN for spatial features and LSTM for sequential patterns
  - **Attention Mechanism**: Optional attention layer for improved performance
  - **Full Pipeline**: 
    - Data loading and preprocessing
    - Model training with progress tracking
    - Interactive testing and prediction
  - **User-Friendly UI**: 
    - Drawing canvas for character input
    - File upload support
    - Real-time visualisation 

## 📦 Installation
### Prerequisites
  - Python 3.8-3.11
  - pip or conda

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/OsamaRabih/arabic-handwriting-recognition.git
   cd arabic-handwriting-recognition
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   """bash
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   """bash

## 🖥️ Usage
### Running Locally
```bash
streamlit run main.py
"""bash

#### Application Workflow
1. Training Page:
  - 📊 Upload CSV datasets (features and labels)
  - ⚙️ Configure model with/without attention
  - 📈 Monitor training progress
2. Testing Page:
  - 🧪 Evaluate model performance
  - 👀 View sample predictions
3. Results Page:
  - 👀 Visualise training metrics →
  - 📤 Export model performance data
4. Prediction Page: 
  - ✍️ Draw characters OR 📤 Upload images
  - 🔮 Get real-time predictions

## 🧠 Model Architecture
  graph TD
      A[32x32 Input Image] --> B[CNN Block]
      B --> C[Max Pooling]
      C --> D[LSTM Layer]
      D --> E{Attention?}
      E -->|Yes| F[Attention Mechanism]
      E -->|No| G[Fully Connected]
      F --> G
      G --> H[28-Class Softmax]
      style A fill:#f9f,stroke:#333
      style H fill:#4CAF50,stroke:#333

## 📂 Project Structure
  arabic-handwriting-recognition/
  ├── classes/
  │   ├── data_handler.py     # Data loading/preprocessing
  │   ├── model_trainer.py    # Model building/training
  │   └── predictor.py        # Prediction logic
  ├── st_pages/
  │   ├── train_page.py       # Training interface
  │   ├── test_page.py        # Testing interface
  │   ├── results_page.py     # Results visualization
  │   ├── predict_page.py     # Prediction interface
  │   └── exit_page.py        # Application exit
  ├── tests/                  # Unit tests
  ├── .streamlit/             # Configuration
  │   └── secrets.toml        # Local secrets
  ├── main.py                 # Main application
  ├── requirements.txt        # Dependencies
  └── README.md               # This file

## 🌐 Streamlit Cloud Deployment
  1.Fork this repository
  2. Go to Streamlit Cloud https://share.streamlit.io/
  3. Click "New app" and connect your GitHub
  4. Set:
    - Repository: OsamaRabih/arabic-handwriting-recognition
    - Branch: main
    - Main file path: main.py
  5. Configure secrets in Settings if needed

## 🧪 Testing
# Run tests
```sh
pytest tests/ -v
# Check coverage
coverage run -m pytest tests/
coverage report -m
"""sh
## 🤝 Contributing
  1. Fork the project
  2. Create your feature branch (git checkout -b feature/AmazingFeature)
  3. Commit your changes (git commit -m 'Add some amazing feature')
  4. Push to the branch (git push origin feature/AmazingFeature)
  5 Open a Pull Request

## 📜 License
Distributed under the MIT License. See LICENSE for more information.

## 📧 Contact
Osama Rabih - rabih.osama91@gmail.com
Project Link: https://github.com/OsamaRabih/arabic-handwriting-recognition





   
