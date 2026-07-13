# Blogify — AI Blog Writing Agent

Blogify is an AI-powered agent that automatically researches, plans, and writes complete technical blog posts on any topic you give it — and can even generate diagrams/images to go along with the content. It uses a **multi-agent workflow** built with **LangGraph**, and comes with a simple, interactive **Streamlit** web interface.

---

## Overview

Instead of writing a blog post from scratch, you simply enter a topic. Blogify's agent pipeline then:

1. Decides whether the topic needs live internet research or can be answered from general knowledge.
2. Gathers evidence from the web (if needed) using Tavily Search.
3. Plans out the blog structure (title, audience, tone, and sections).
4. Writes each section in parallel using "worker" agents.
5. Merges all sections into one polished blog post.
6. Optionally generates and places relevant diagrams/images using Google Gemini.
7. Lets you preview, download, and manage all your generated blogs — right from the browser.

---

## Features

- Smart Research Routing — automatically detects whether a topic needs research (e.g. "Best GPUs in 2026") or can be written from existing knowledge (e.g. "Explain Binary Search").
- Multi-Agent Pipeline — built with LangGraph, using a Router → Research → Orchestrator → Workers → Reducer architecture.
- Live Web Research — pulls up-to-date evidence using the Tavily Search API and cites sources in the blog.
- Parallel Section Writing — multiple blog sections are drafted simultaneously for faster generation.
- AI Image Generation (optional) — automatically decides where diagrams/images would help and generates them using Google Gemini.
- Interactive Streamlit UI — enter a topic, watch the agent work in real time, and preview the final blog.
- Blog History — browse, reload, and re-download previously generated blogs.
- Export Options — download the blog as Markdown, or as a zipped bundle with images included.
- Dockerized — run the entire app with a single `docker compose up` command.

---

## Architecture

Blogify's backend is built as a **state graph** using LangGraph:

```
START
  │
  ▼
Router ──(needs research?)──► Research (Tavily)
  │                                   │
  └──────────────►  Orchestrator  ◄───┘
                        │
                (creates section plan)
                        │
                        ▼
                 Worker (x N, parallel)
                        │
                        ▼
                 Reducer subgraph:
             merge_content → decide_images → generate_and_place_images
                        │
                        ▼
                       END
```

- Router – Classifies the topic as `closed_book`, `hybrid`, or `open_book` to decide if research is required.
- Research – Runs web searches via Tavily and extracts clean, structured evidence.
- Orchestrator – Creates a section-by-section content plan (the blog "outline").
- Worker – Each section is written independently and in parallel by an LLM.
- Reducer – Merges all sections into the final blog, decides if images are needed, generates them, and saves the final Markdown file.

---

## Tech Stack

| Layer            | Technology                                   |
|-------------------|-----------------------------------------------|
| Frontend / UI     | [Streamlit](https://streamlit.io/)            |
| Agent Orchestration | [LangGraph](https://www.langchain.com/langgraph) + [LangChain](https://www.langchain.com/) |
| LLM (text)        | [Ollama](https://ollama.com/) running `llama3.2` locally |
| Web Research       | [Tavily Search API](https://tavily.com/)       |
| Image Generation   | [Google Gemini](https://ai.google.dev/) (`gemini-2.5-flash-image`) |
| Data Validation    | [Pydantic](https://docs.pydantic.dev/)         |
| Language           | Python 3.12                                    |
| Containerization   | Docker & Docker Compose                        |

---

## Project Structure

```
Blogify-main/
├── bwa_backend.py       # LangGraph agent pipeline (router, research, orchestrator, workers, reducer)
├── bwa_frontend.py       # Streamlit web interface
├── requirements.txt      # Python dependencies
├── dockerfile             # Docker image definition
├── docker-compose.yml    # Docker Compose service configuration
├── generated_blogs/      # Auto-created: stores generated blog Markdown files
└── images/                # Auto-created: stores AI-generated images
```

---

## Prerequisites

Before running Blogify, make sure you have:

- Python 3.12+ installed
- [Ollama](https://ollama.com/) installed and running locally, with the `llama3.2` model pulled:
  ```bash
  ollama pull llama3.2
  ```
- A Tavily API key (required for web research) → [Get one here](https://tavily.com/)
- A Google Gemini API key (optional, only needed if you want AI-generated images) → [Get one here](https://ai.google.dev/)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Blogify.git
cd Blogify-main
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
TAVILY_API_KEY=your_tavily_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Run the app

```bash
streamlit run bwa_frontend.py
```

Then open your browser at **http://localhost:8501**

---

## Running with Docker

If you'd rather skip local setup, run Blogify in a container:

```bash
docker compose up --build
```

This will build the image and start the app at **http://localhost:8501**. Make sure your `.env` file is present in the project root before running — Docker Compose reads it automatically.

---

## How to Use

1. Open the app in your browser.
2. Enter a topic in the sidebar (e.g. "Introduction to REST APIs").
3. Select the "as-of date" and choose whether to enable "AI image generation".
4. Click  "Generate Blog" and watch the pipeline run in real time.
5. Once complete, explore the tabs:
   - Plan – see the generated outline and section breakdown
   - Evidence – view the research sources used (if any)
   - Preview – read the final blog with images rendered inline
   - Images – check the status of each generated image
   - Logs – view detailed pipeline execution logs
6. Download the blog as Markdown, or as a zipped bundle with images.

---

## Future Improvements

- Support for additional LLM providers (OpenAI, Anthropic, etc.)
- User authentication and multi-user blog history
- SEO metadata generation (titles, descriptions, tags)
- One-click publishing to platforms like Medium or Dev.to
- Support for exporting blogs as PDF/HTML

---

## Acknowledgements

Built as a learning project to explore multi-agent orchestration with LangGraph, combining research, planning, parallel content generation, and image generation into a single automated workflow.

---

## License

This project is open source and available for learning and personal use. Feel free to fork and build on it!
