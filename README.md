# AI Classroom Monitoring System Documentation

## Overview
The AI Classroom Monitoring System is an advanced Artificial Intelligence-based framework designed to monitor and evaluate student engagement, attentiveness, and physiological alertness in classroom environments. It integrates Computer Vision, Deep Learning, Learning Analytics, and Remote Photoplethysmography (rPPG) to provide real-time analytics and decision support for educators.

## Features
### Student Management
- Registration and profile management
- Performance tracking
- Attention ranking and analytics
- Engagement monitoring

### Teacher Analytics
- Teacher registration
- Teaching effectiveness assessment
- Classroom performance evaluation
- AI-generated recommendations

### Classroom Monitoring
- Supports image, video, live webcam, and CCTV/RTSP stream analysis
- Behaviour detection (e.g., attentive, distracted, sleepy, phone use)

### Physiological Monitoring
- Pulse signal extraction
- Heart rate estimation
- Physiological alertness assessment
- Pulse waveform visualization

## System Architecture
### Methodology
- **Computer Vision Layer**: YOLO-based behaviour detection, OpenCV for image/video processing
- **Learning Analytics Layer**: Attention scoring, engagement evaluation, behaviour classification
- **Physiological Computing Layer**: rPPG for signal acquisition, filtering, BPM estimation, and alertness computation
- **Decision Fusion Layer**: Combines attention and physiological alertness scores for comprehensive classroom performance evaluation

## Technology Stack
- **Programming Language**: Python
- **Dashboard Framework**: Streamlit
- **Computer Vision**: OpenCV
- **Deep Learning**: YOLO (Ultralytics)
- **Signal Processing**: NumPy
- **Data Analysis**: Pandas
- **Visualization**: Plotly
- **Deployment Platform**: Render
- **Version Control**: GitHub

## Installation and Setup
1. Clone the repository:
    ```shell
    git clone <repository-url>
    cd attention_classroom_monitoring
    ```
2. Create and activate a virtual environment:
    ```shell
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```
3. Install dependencies:
    ```shell
    pip install -r requirements.txt
    ```

## Running the Application
- Launch the application:
  ```shell
  streamlit run dashboard.py
  ```
- Access the dashboard at:
  ```
  http://localhost:8501
  ```

## Deployment
- Cloud deployment via Render:
  ```shell
  streamlit run dashboard.py \
  --server.address=0.0.0.0 \
  --server.port=$PORT
  ```

## Outputs
- Attention Score
- Alertness Score
- Final Classroom Score
- Heart Rate Estimation (BPM)
- Pulse Waveform
- Student Attention Ranking
- Behaviour Distribution Analysis
- Historical Attention Trends
- Teacher Performance Metrics
- Engagement Recommendations

## Future Enhancements
- Smartwatch integration
- QR code-based device pairing
- Emotion recognition
- Automated attendance management
- Multi-camera monitoring
- Mobile application development
- Real-time notification system
- Cloud-based learning analytics platform

## Author
- **Name**: Happy Oseretin
- **Affiliation**: Department of Computer Science
- **Research Area**: Artificial Intelligence, Computer Vision, Learning Analytics, Educational Technology, Human-Centered Computing

## License
This project is intended for academic research and educational purposes. Redistribution or commercial use requires the author's approval.