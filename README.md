# 🎙️ AI Meeting Intelligence — Real & Synthetic Meeting Analyzer

An AI-powered meeting intelligence system that converts **real meeting audio or synthetic meeting topics into structured meeting minutes** using open-source AI models.

The project combines **Whisper** for speech-to-text, **Llama 3.2 3B Instruct** for meeting generation and summarization, Hugging Face pipelines for additional analysis, and **Gradio** for an interactive interface.

---

## 🚀 Overview

AI Meeting Intelligence provides two ways to analyze meetings:

### 🎧 Real Meeting Mode

Upload a meeting audio file.

```text
Audio File
    ↓
Whisper
    ↓
Transcript
    ↓
Llama 3.2 3B Instruct
    ↓
Structured Meeting Minutes
```

### 🧪 Synthetic Meeting Mode

Enter a meeting topic and allow Llama to generate a fictional meeting conversation.

```text
Meeting Topic
    ↓
Llama 3.2
    ↓
Synthetic Meeting Transcript
    ↓
Llama 3.2
    ↓
Structured Meeting Minutes
```

The two workflows ultimately produce structured meeting information through the same analysis pipeline.

---

## ✨ Features

* 🎧 Upload and transcribe real meeting audio
* 🧪 Generate realistic synthetic meeting conversations
* 📝 Automatically generate structured meeting minutes
* 📌 Extract discussion points and takeaways
* ✅ Identify action items and owners when clearly available
* 😊 Analyze meeting sentiment
* 🏷️ Classify meetings into predefined categories
* ⚡ GPU-accelerated inference
* 🧠 Run Llama 3.2 using 4-bit quantization
* 🖥️ Interactive Gradio interface
* 🔐 Hugging Face authentication through Google Colab Secrets

The final interface exposes transcript, meeting minutes, sentiment, and meeting category information.

---

## 🏗️ System Architecture

```text
                    AI MEETING INTELLIGENCE
                             │
                ┌────────────┴────────────┐
                │                         │
          REAL MEETING             SYNTHETIC MEETING
                │                         │
          Audio File                    Topic
                │                         │
                ▼                         ▼
             Whisper                   Llama 3.2
                │                         │
                ▼                         ▼
           Transcript            Synthetic Transcript
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
                       Llama 3.2 3B
                             │
                             ▼
                    Meeting Analysis
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
          Summary       Discussion       Action Items
                             │
                             ▼
                       Takeaways
                             │
                             ▼
                         Gradio UI
```

---

## 🛠️ Technology Stack

| Technology                    | Purpose                         |
| ----------------------------- | ------------------------------- |
| **Python**                    | Core programming language       |
| **PyTorch**                   | Deep learning inference         |
| **Hugging Face Transformers** | Model loading and inference     |
| **Whisper**                   | Automatic speech recognition    |
| **Llama 3.2 3B Instruct**     | Meeting generation and analysis |
| **BitsAndBytes**              | 4-bit quantization              |
| **Hugging Face Pipelines**    | Sentiment and classification    |
| **Gradio**                    | Interactive web interface       |
| **Google Colab**              | GPU-based execution environment |

The project specifically uses `openai/whisper-medium.en` and `meta-llama/Llama-3.2-3B-Instruct`.

---

## ⚙️ Requirements

The project is designed to run in **Google Colab** with GPU acceleration.

A CUDA-compatible GPU is recommended because both Whisper and the quantized Llama model use GPU inference. The documented test environment used:

```text
PyTorch: 2.11.0+cu128
GPU: Tesla T4
CUDA: Available
```

---

## 📦 Installation

Install the required packages:

```bash
pip install -q --upgrade bitsandbytes accelerate transformers==4.57.6 gradio
```

The project uses these packages for Transformers, quantization, GPU acceleration, and the Gradio interface.

---

## 🔐 Hugging Face Authentication

The Llama model is hosted on Hugging Face and requires authentication.

In Google Colab:

1. Open the **Secrets** panel.
2. Add a new secret named:

```text
HF_TOKEN
```

3. Paste your Hugging Face access token.
4. Enable notebook access for the secret.

The project retrieves the token through Colab's `userdata` mechanism rather than hard-coding credentials in the notebook.

> **Security:** Never commit your Hugging Face token to GitHub.

---

## 🤖 Models

### Whisper

```text
openai/whisper-medium.en
```

**Purpose:** Automatic speech recognition.

```text
Meeting Audio → Whisper → Transcript
```

Whisper is used to convert uploaded meeting audio into text.

### Llama 3.2 3B Instruct

```text
meta-llama/Llama-3.2-3B-Instruct
```

**Purpose:**

* Generate synthetic meetings
* Generate structured meeting minutes

The model is loaded with 4-bit quantization to reduce GPU memory usage.

---

## ⚡ 4-Bit Quantization

The Llama model uses **BitsAndBytes 4-bit quantization**.

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```

Quantization reduces the numerical precision used to represent model weights, allowing the 3B parameter model to run more efficiently on a Colab GPU.

---

# 🎧 Real Meeting Analysis

The real meeting workflow accepts an audio file as input.

```text
Audio File
     ↓
Whisper
     ↓
Transcript
     ↓
Llama 3.2
     ↓
Meeting Minutes
```

The documented implementation uses the Hugging Face `pipeline()` API for automatic speech recognition.

Example:

```python
result = whisper_pipe(audio_filename)
transcription = result["text"]
```

The resulting transcription is then passed to the meeting-minutes generator.

---

# 🧪 Synthetic Meeting Generation

Instead of uploading audio, users can provide a meeting topic.

Example:

```text
Planning a university AI hackathon
```

Llama generates a fictional professional meeting containing:

* 3–4 participants
* Different opinions
* Suggestions
* At least one decision
* At least two action items

The generated conversation is then passed through the same meeting-minutes generation pipeline.

Example workflow:

```text
Topic
 ↓
Llama 3.2
 ↓
Synthetic Meeting
 ↓
Llama 3.2
 ↓
Meeting Minutes
```

---

# 📝 Meeting Minutes Generation

The system instructs Llama to create structured meeting minutes from the transcript.

The intended output contains:

```text
## Summary

## Discussion Points

## Takeaways

## Action Items
```

The prompt explicitly instructs the model to:

* Use only information available in the transcript
* Avoid inventing names, dates, locations, attendees, or facts
* Keep the summary concise
* Avoid repeating the transcript

---

# 📊 Additional Meeting Analysis

## 😊 Sentiment Analysis

The project uses a Hugging Face sentiment-analysis pipeline to analyze the generated meeting transcript.

Example output:

```text
POSITIVE: 99.45%
```

---

## 🏷️ Meeting Category Classification

The project also performs zero-shot classification using predefined categories:

```python
categories = [
    "Technology",
    "Education",
    "Business",
    "Project Management",
    "Finance",
    "Other"
]
```

For the documented synthetic meeting example, the highest predicted category was:

```text
Education — 65.93%
```

---

# 🖥️ Gradio Interface

The application provides two main tabs:

### 🎧 Real Meeting

```text
Upload Meeting Audio
        ↓
Analyze Meeting
        ↓
Transcript
        ↓
Meeting Minutes
```

### 🧪 Synthetic Meeting

```text
Enter Meeting Topic
        ↓
Generate Meeting
        ↓
Synthetic Transcript
        ↓
Meeting Minutes
```

The Gradio interface is designed to display the generated transcript and meeting minutes interactively.

---

## 📋 Example Output

For a meeting transcript, the system generates structured information such as:

```markdown
## Summary

The meeting discussed the key topics and decisions
identified from the provided transcript.

## Discussion Points

- Main topic discussed
- Important suggestions
- Key concerns

## Takeaways

- Important conclusion
- Main decision

## Action Items

- Owner: Assigned task
- Owner: Follow-up task
```

The project demonstrates this workflow using both a real Denver City Council transcript and a synthetic university AI hackathon meeting.

---

# 🔄 Complete Workflow

```text
                     USER INPUT
                         │
              ┌──────────┴──────────┐
              │                     │
         REAL AUDIO              TOPIC
              │                     │
              ▼                     ▼
           WHISPER              LLAMA 3.2
              │                     │
              ▼                     ▼
        TRANSCRIPT          SYNTHETIC TRANSCRIPT
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                   LLAMA 3.2 3B
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Summary    Discussion   Takeaways
                         │
                         ▼
                    Action Items
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
        Sentiment     Category     Gradio
```

---

# 📁 Suggested Repository Structure

```text
AI-Meeting-Intelligence/
│
├── README.md
├── AI_Meeting_Intelligence.ipynb
├── requirements.txt
├── .gitignore
│
├── data/
│   └── sample_audio/
│
├── screenshots/
│   └── gradio-interface.png
│
└── docs/
    └── project-notes.md
```

> The exact repository structure is not specified in the source notebook; the structure above is a recommended organization for publishing the project on GitHub.

---

# ▶️ Running the Project

### 1. Open the notebook

Open the project in Google Colab.

### 2. Enable GPU

In Colab:

```text
Runtime
→ Change runtime type
→ Hardware accelerator
→ GPU
```

### 3. Install dependencies

```bash
pip install -q --upgrade bitsandbytes accelerate transformers==4.57.6 gradio
```

### 4. Configure Hugging Face

Add:

```text
HF_TOKEN
```

to Colab Secrets.

### 5. Load the models

The notebook loads:

```text
Whisper Medium English
Llama 3.2 3B Instruct
```

### 6. Choose a mode

Use either:

```text
Real Meeting
```

or:

```text
Synthetic Meeting
```

### 7. Analyze

The application generates the transcript and structured meeting minutes.

---

# 🔒 Privacy & Security Considerations

This project processes meeting transcripts using AI models.

For real-world deployment:

* Do not commit API/access tokens.
* Avoid uploading confidential meetings to public services.
* Consider local/private model deployment for sensitive meetings.
* Review generated summaries before using them as official meeting records.

The synthetic meeting generator is explicitly designed to create fictional conversations and avoid real personal information.

---

# 🎯 Learning Objectives

This project demonstrates practical implementation of:

* Automatic Speech Recognition
* Large Language Models
* Hugging Face Transformers
* Hugging Face Pipelines
* Chat Templates
* Tokenization
* 4-bit Quantization
* GPU-accelerated inference
* Synthetic data generation
* Text summarization
* Sentiment analysis
* Zero-shot classification
* Interactive AI applications with Gradio

---

# 🚀 Future Improvements

Potential extensions include:

* Speaker diarization
* Speaker identification
* More accurate action-item extraction
* Meeting title generation
* Key-decision extraction
* Topic segmentation
* Multi-language transcription
* Multi-language meeting summaries
* Persistent meeting history
* Search across previous meetings
* Export meeting minutes to PDF/Word
* Calendar integration
* Authentication and user accounts
* Production deployment
* Private/local model inference

These are proposed extensions rather than features documented as currently implemented.

---

# 📌 Project Highlights

| Component          | Implementation                  |
| ------------------ | ------------------------------- |
| Speech Recognition | Whisper Medium English          |
| LLM                | Llama 3.2 3B Instruct           |
| Quantization       | 4-bit NF4                       |
| Framework          | Hugging Face Transformers       |
| Sentiment          | Hugging Face Sentiment Pipeline |
| Classification     | Zero-Shot Classification        |
| UI                 | Gradio                          |
| Environment        | Google Colab                    |
| Acceleration       | CUDA GPU                        |

---

# 👨‍💻 Project Summary

**AI Meeting Intelligence** demonstrates how open-source AI models can be combined into an end-to-end meeting analysis pipeline.

Instead of relying on a single model, the system combines:

```text
Whisper
   +
Llama 3.2
   +
Hugging Face Pipelines
   +
Gradio
```

to transform meeting audio or a meeting topic into useful, structured meeting intelligence.

The project showcases practical skills in **LLM application development, speech recognition, model quantization, NLP pipelines, synthetic data generation, and AI-powered UI development**.

---

## ⭐ If You Find This Project Useful

Consider giving the repository a ⭐ on GitHub and exploring the implementation.

---

## 📄 License

Add the license appropriate for the models, code, and datasets used in your final repository. Model-specific licensing should be checked before redistribution or commercial deployment.
