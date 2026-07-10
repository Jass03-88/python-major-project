# NextGen Biometric Attendance & Security System

A high-performance, edge-AI face recognition and attendance tracking application built in Python. This system replaces outdated Haar cascades with state-of-the-art Deep Learning models (SFace & YuNet) for blazing-fast recognition, incorporates anti-spoofing liveness detection via MediaPipe, and features a completely multi-threaded modern GUI for a buttery-smooth user experience.

## ✨ Features

- **Deep Learning Core**: Utilizes OpenCV's **YuNet** for highly accurate face detection and **SFace** for sub-millisecond facial embedding extraction and recognition.
- **Liveness & Anti-Spoofing**: Integrates **MediaPipe's FaceLandmarker Task API** to track Eye Aspect Ratios (EAR) and detect physical blinks, ensuring high security against spoofing attacks (photos/screens).
- **Automated Intruder Alerts**: Silently captures photos of unknown faces (or failed liveness checks) and transmits them instantly via a **Telegram Bot** integration.
- **Multi-Threaded Architecture**: Camera frame polling and deep learning inference are completely offloaded to background worker threads, keeping the UI perfectly responsive with zero freezing.
- **Premium Dark UI**: Built with `ttkbootstrap` for a stunning, modern dark-mode aesthetic complete with hover effects, tooltips, and dynamic status bars.
- **Data Analytics Dashboard**: Auto-generates clean CSV reports and renders graphical attendance analytics via Matplotlib directly inside the application.

## 🛠️ Prerequisites

- **Python 3.8+**
- A working webcam
- A Telegram Bot Token and Chat ID (if you wish to use the intruder alert feature)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/biometric-attendance.git
   cd biometric-attendance
   ```

2. **Set up a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Telegram bot credentials:
   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

*(Note: The required MediaPipe Task model will automatically download on the first run).*

## 🚀 Usage

Launch the main application by running:

```bash
python gui.py
```

### Application Flow:
1. **Admin Setup**: Set up your master password to protect sensitive operations.
2. **Registration**: Register authorized users. The system will capture multiple angles of their face to train the embeddings map.
3. **Face Login**: Users stand in front of the camera and blink. If verified, they are successfully checked in for the day. If an unknown face attempts access, a snapshot is forwarded to the Telegram bot.

## 📂 Project Structure

```text
├── gui.py                  # Main application entry point and UI layout
├── recognition_core.py     # Inference engine orchestrating SFace embeddings
├── face_detector.py        # YuNet Face Detection wrapper
├── liveness_utils.py       # MediaPipe EAR blink tracking logic
├── attendance_manager.py   # SQLite database operations and CSV exports
├── telegram_utils.py       # Telegram bot HTTP API integration
├── security_utils.py       # Master password hashing and brute-force lockout
├── dnn_face_detector/      # Stores SFace/YuNet ONNX models & FaceLandmarker tasks
├── dataset/                # Captured face crops for registered users
└── .env                    # Secret configurations
```

## 🛡️ Disclaimer
This software is built for educational and internal organizational use. Ensure you comply with local privacy and biometric data storage laws before deploying it in a public setting.
