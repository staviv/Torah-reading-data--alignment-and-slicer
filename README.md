
# Torah Reading Data Alignment for Training

[עברית](README.he.md)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge&logo=pytorch" alt="Framework">
</p>

<p align="center">
  An automated pipeline to create high-quality, time-aligned datasets for training ASR models for Torah reading with cantillation marks (<em>te'amim</em>).
</p>

This project, developed by Aviv Shem-Tov, addresses a critical bottleneck in developing speech recognition models for biblical texts: the lack of large, accurately segmented, and time-aligned training data. We developed a comprehensive, automated system that transforms long audio recordings of Torah readings into short, precisely aligned segments suitable for training models like Whisper.

The system successfully expanded our training dataset from 123 to **337 hours**, incorporating a diverse range of cantillation styles and readers. This led to a dramatic improvement in model performance, with the F1 score for Ashkenazi cantillation recognition increasing from **0.153 to 0.842** and the Word Error Rate (WER) dropping from **91% to a mere 7.3%**.

---

## 📜 Table of Contents

- [Key Features](#-key-features)
- [The Problem](#-the-problem)
- [The Automated Pipeline](#-the-automated-pipeline)
- [Results & Impact](#-results--impact)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#-configuration)
- [Usage](#-usage)
  - [1. Data Collection from YouTube](#1-data-collection-from-youtube)
  - [2. Main Processing Pipeline](#2-main-processing-pipeline)
  - [3. Analyzing the Dataset](#3-analyzing-the-dataset)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Key Features

- **🤖 Automated YouTube Sourcing**: Uses a Gemini-powered LLM to automatically find, identify (Parasha, Aliyah, and Nusach), and download relevant Torah reading videos and playlists.
- **🗣️ VAD-Based Smart Segmentation**: Intelligently splits long audio recordings into segments under 30 seconds by detecting natural pauses, avoiding cuts in the middle of words.
- **🔄 Iterative Text Alignment**: A novel, multi-step process that gradually aligns the initial Whisper transcription with the complex, fully-vocalized biblical text, bridging the gap between modern and traditional Hebrew.
- **⚡ Multi-GPU Batch Processing**: Optimized for speed, the pipeline leverages multiple GPUs and batch processing to transcribe and align data efficiently.
- **📊 Data Quality & Analysis Tools**: Includes scripts to analyze SRT files for quality control, identify outliers, and remove common transcription hallucinations (e.g., "תודה רבה").
- **🗂️ Structured Data Management**: Organizes raw and processed data into a clean, predictable directory structure for easy management and reproducibility.

---

## 🎯 The Problem

Training modern ASR models like Whisper requires short audio clips (typically under 30 seconds) with precise transcriptions. However, most available recordings of Torah readings are long-form, covering entire sections (*aliyot*) or weekly portions (*parshiot*). Manually segmenting these long recordings and aligning them with the Masoretic text—which includes complex vowelization (*nikud*) and cantillation marks (*te'amim*)—is an extremely slow, tedious, and error-prone process. This data scarcity was the primary obstacle to improving Torah reading recognition models.

---

## 🛠️ The Automated Pipeline

Our solution is an end-to-end pipeline that automates the entire data preparation process. The architecture is as follows:

```mermaid
graph TD
    subgraph "Input"
        A[YouTube Link / Playlist]
        B[Local Audio File]
    end

    subgraph "Processing Pipeline"
        A --> C{Identify Parasha/Aliyah via LLM};
        C --> D[Download Audio using yt-dlp];
        D --> E[Audio Segmentation using VAD];
        B --> E;
        E --> F[Initial Transcription & Raw SRT Generation];
        F --> G[Iterative Synchronization];
    end

    subgraph "Output"
        G --> H[Final Aligned SRT (Training-Ready Dataset)];
    end
```

1.  **Data Ingestion**: The pipeline can start from a **YouTube link** or a **local audio file**.
    -   If a YouTube link is provided, an LLM (Gemini) analyzes its metadata to identify the content, and `yt-dlp` downloads the audio.
2.  **Audio Segmentation (`audio_to_srt.py`)**: The long audio file is segmented into smaller clips using an energy-based Voice Activity Detection (VAD) algorithm, ensuring cuts are made during natural pauses.
3.  **Initial Transcription (`audio_to_srt.py`)**: Each short clip is passed through a fine-tuned Whisper model to generate a "raw" transcription (`_RAW.srt`), which has accurate timings but a simplified text.
4.  **Iterative Text Alignment (`main.py`)**: This is the core innovation. Instead of a single, direct alignment, we use `WhisperTimeSync.jar` to gradually align the raw SRT with progressively more complex versions of the biblical text, until a perfect alignment with the fully vocalized and cantillated text is achieved.
5.  **Final Output & Storage**: The final output is a perfectly timed SRT file with the accurate Masoretic text, which serves as the training-ready dataset.

---

## 📈 Results & Impact

The automated pipeline enabled a massive expansion of our dataset and a corresponding leap in model performance.

**Dataset Expansion:**

| Dataset | Before | After | Increase |
| :-------- | :----: | :---: | :------: |
| **Hours** | 123 | **337** | **+174%** |

**Model Performance (Ashkenazi Nusach - `V3-turbo` model):**

| Metric | Before (Base Dataset) | After (Expanded Dataset) | Improvement |
| :----------- | :-------------------: | :----------------------: | :---------: |
| **F1 Score** | 0.153 | **0.842** | **+450%** |
| **WER** | 91.0% | **7.3%** | **-92%** |

These results confirm the system's effectiveness in generating high-quality data and significantly enhancing Torah reading recognition capabilities.

---

## 📂 Repository Structure

```
.
├── automatic using WhisperTimeSync/
│   ├── main.py                   # Main orchestrator for the processing pipeline
│   ├── audio_to_srt.py           # Handles VAD segmentation and initial Whisper transcription
│   ├── get_data_from_youtube.py  # Automated data collection from YouTube using LLM
│   ├── create_steps_files.py     # Generates intermediate text files for iterative alignment
│   ├── nikud_and_teamim.py       # Utilities for processing Hebrew text (nikud, teamim)
│   ├── analyze_srt.py            # Utility script for dataset analysis and quality control
│   ├── parasha_matcher.py        # Matches and standardizes Parasha names
│   └── WhisperTimeSync/          # Contains the external Java alignment tool
│
├── semi automatic cut/
│   └── cut audio.py              # Legacy semi-automatic GUI tool for segmentation
│
├── delete_files_with_hilucinations.py # Script to clean dataset from Whisper hallucinations
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Java** (required to run `WhisperTimeSync.jar`)
- **FFmpeg** (required by `yt-dlp` and `librosa`)
- A **Google Gemini API Key** for the automated data collection script.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install torch transformers datasets librosa soundfile srt tqdm google-generativeai yt-dlp
    ```

3.  **Set up API Key:**
    Set your Gemini API key as an environment variable:
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    ```

4.  **WhisperTimeSync:**
    Ensure the `WhisperTimeSync.jar` file is located in the `automatic using WhisperTimeSync/WhisperTimeSync/distrib/` directory.

### ⚙️ Configuration

**Important:** Before running the scripts, you must configure the file paths to match your local environment. The paths are currently hard-coded in the scripts.

-   In `automatic using WhisperTimeSync/main.py`, modify the following path variables at the top of the file:
    -   `base_dir`
    -   `text_dir`
    -   `raw_srt_storage_base`
    -   `whisperTimeSync`
-   In `automatic using WhisperTimeSync/get_data_from_youtube.py`, modify the `dataset_dir` path inside the `process_single_video` function and other relevant functions.
-   In `delete_files_with_hilucinations.py`, modify `raw_srt_base_dir` and `base_dir` at the bottom of the script.

---

## Usage

### 1. Data Collection from YouTube

The `get_data_from_youtube.py` script can be run to download new audio data. It supports interactive and automatic modes.

-   **Interactive Mode**: Guides you through processing a single video or playlist, with prompts for confirmation.
    ```bash
    python "automatic using WhisperTimeSync/get_data_from_youtube.py"
    ```
-   **Automatic Mode**: Processes entire playlists or a file of links without user interaction, ideal for large-scale data collection.

### 2. Main Processing Pipeline

The `main.py` script is the core of the project. It automatically finds audio files that haven't been processed and runs the full alignment pipeline.

1.  After configuring the paths, place your audio files in their respective dataset folders inside your configured `base_dir`.
2.  Place the corresponding text files inside your configured `text_dir`.
3.  **Run the main script:**
    ```bash
    python "automatic using WhisperTimeSync/main.py"
    ```
    The script will process all new files, skip existing ones, and save the final aligned SRTs in the same directory as the audio files.

### 3. Analyzing the Dataset

Use `analyze_srt.py` to inspect the quality of your generated dataset.

```bash
# Example: Find files with subtitles longer than 150 characters
python "automatic using WhisperTimeSync/analyze_srt.py" --dir <your_base_dir>/MyDataset/ --recursive --length-threshold 150
```

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

---

## 🙏 Acknowledgments

- **OpenAI** for the Whisper model.
- **Google** for the Gemini model.
- **Sefaria** for providing the essential biblical texts via their API.
- The **yt-dlp** team for their powerful video downloading tool.
- **EtienneAb3d** for the [WhisperTimeSync](https://github.com/EtienneAb3d/WhisperTimeSync) tool, which was a key component of our alignment process.
