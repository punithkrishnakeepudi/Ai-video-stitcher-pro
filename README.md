
# AI Video Stitcher Pro

🎬 AI Video Stitcher Pro is a powerful web-based application designed to combine multiple videos into a single output using advanced stitching techniques. Built with Flask for the backend and a user-friendly HTML interface for the frontend, it leverages OpenCV and optional machine learning libraries (scikit-learn, TensorFlow) to provide seamless video stitching with features like AI-enhanced panorama stitching, frame normalization, and audio integration.

---

## 🚀 Features

### Multiple Stitching Methods:
- **Horizontal**: Combines videos side by side.
- **Vertical**: Stacks videos vertically.
- **Grid**: Arranges videos in a grid layout.
- **AI Panorama**: Uses intelligent stitching with OpenCV's Stitcher for seamless results.

### Seamless Blending:
- Reduces visible seams with advanced blending techniques.

### AI Enhancements *(optional)*:
- Frame quality improvement using denoising, sharpening, and contrast enhancement.
- Intelligent feature matching for panorama stitching.

### Preview Generation:
- View stitched frame previews before final processing.

### Flexible Output Options:
- **Resolutions**: 720p, 1080p, 2K, 4K, or original quality.
- **Formats**: MP4, AVI, MOV, MKV.

### Additional Features:
- 🎧 Audio Support
- ⏱ Auto-Sync videos by audio
- 📊 Progress Tracking
- 🧹 File Management and automatic cleanup

---

## 🧰 Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)
- Git

### System Dependencies:

#### Linux:
- `python3-pip`, `python3-venv`, `ffmpeg`, `libgl1-mesa-glx`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`, `libgomp1`

#### macOS:
- Homebrew, Python 3, FFmpeg

#### Windows:
- Python 3.8+, FFmpeg, Visual Studio Build Tools

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/punithkrishnakeepudi/ai-video-stitcher-pro.git
cd ai-video-stitcher-pro
```


### 2. Run the Setup Script

```bash
python3 setup.py
# or on Windows:
python setup.py
```

This will:
- Check Python version
- Install system dependencies
- Create a virtual environment
- Install Python packages
- Setup folders like `uploads`, `outputs`, etc.
- Create run scripts (`run.sh` or `run.bat`)

### 3. Activate the Virtual Environment

```bash
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 4. Install Optional ML Dependencies

```bash
pip install scikit-learn==1.3.0 tensorflow==2.13.0
```

### 5. Verify FFmpeg

```bash
ffmpeg -version
```

---

## 🧪 Usage

### Running the App

```bash
# Linux/macOS:
./run.sh

# Windows:
run.bat

# Or manually:
python3 app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

### Using the Web Interface

1. Upload **2–6 videos** (MP4, AVI, MOV, MKV)
2. Select:
   - Stitching method
   - Output quality
   - Include audio
   - AI enhancement
   - Output format
3. Click **"Generate Preview"**
4. Confirm & Process
5. Download the final video

---

## 📡 API Endpoints

| Endpoint               | Method | Description                           |
|------------------------|--------|---------------------------------------|
| `/process`             | POST   | Upload videos & generate preview      |
| `/start_processing/<id>` | POST | Begin processing after preview        |
| `/progress/<id>`       | GET    | Check status                          |
| `/preview/<filename>`  | GET    | Get preview images                    |
| `/download/<filename>` | GET    | Download final video                  |
| `/cleanup`             | GET    | Clean up old files                    |
| `/health`              | GET    | Server status                         |

### Example `curl` Command:

```bash
curl -X POST -F "video_1=@video1.mp4" -F "video_2=@video2.mp4" -F "method=horizontal" -F "quality=1080p" -F "format=mp4" http://localhost:5000/process
```

---

## 📁 Project Structure

```
ai-video-stitcher-pro/
├── app.py              # Flask backend
├── index.html          # Frontend interface
├── setup.py            # Setup script
├── requirements.txt    # Python dependencies
├── run.sh / run.bat    # Run scripts
├── uploads/            # Uploaded videos
├── outputs/            # Processed videos
├── previews/           # Preview frames
├── temp/, static/      # Supporting folders
└── venv/               # Virtual environment
```

---

## 🧩 Troubleshooting

- **FFmpeg Not Found**: Ensure it’s installed and in PATH.
- **ML Features Not Working**: Install `scikit-learn` and `tensorflow`.
- **Large File Error**: Limit is 100MB/file and 500MB total.
- **Preview Timeout**: Extend timeout in `app.py`.
- **Install Issues**: Try `pip install -r requirements.txt` inside the venv.

---

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch
3. Commit your changes
4. Push and create a pull request

---

## 📄 License

MIT License. See `LICENSE`.

---

## 🙏 Acknowledgments

- Built with Flask, OpenCV, FFmpeg
- AI support via TensorFlow and scikit-learn
- Inspired by the need for seamless video merging with AI enhancement
