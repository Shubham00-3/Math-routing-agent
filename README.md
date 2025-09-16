# Math Routing Agent

[cite\_start]The Math Routing Agent is a sophisticated Agentic-RAG (Retrieval-Augmented Generation) system designed to function as an AI-powered mathematical professor[cite: 4]. It understands mathematical questions and provides detailed, step-by-step solutions tailored for students. [cite\_start]The system leverages a knowledge base, web search capabilities, and a human-in-the-loop feedback mechanism to continuously improve its accuracy and clarity[cite: 24, 25, 26].

## Core Features

  * **Agentic RAG Pipeline:** The system first queries a Qdrant vector database for existing solutions. [cite\_start]If no relevant information is found, it automatically performs a web search to gather context before generating an answer[cite: 6, 7].
  * [cite\_start]**AI Gateway & Guardrails:** Implements robust input and output guardrails to ensure all content is strictly mathematical and educational, filtering out irrelevant or inappropriate queries[cite: 10, 11].
  * **Human-in-the-Loop Feedback:** Users can provide feedback on the quality of solutions. [cite\_start]This feedback is processed by a dedicated feedback agent to identify areas for improvement[cite: 23].
  * [cite\_start]**DSPy-Powered Self-Learning:** Utilizes the DSPy library to analyze aggregated user feedback and automatically fine-tune the language model, allowing the agent to learn and refine its responses over time (Bonus Requirement Met)[cite: 27, 54].
  * [cite\_start]**MCP for Web Search:** Uses the Model Context Protocol (MCP) to interact with a dedicated web search server, ensuring a modular and scalable architecture for tool use[cite: 22, 53].

## Architecture

The application follows a modern, decoupled architecture:

  * **Frontend:** A responsive web interface built with **React (Next.js)** allows users to interact with the agent.
  * **Backend:** A powerful **FastAPI** server orchestrates the agentic workflow, handles API requests, and manages WebSocket connections for real-time communication.
  * **Infrastructure:** A **Docker Compose** setup manages the necessary services, including a **PostgreSQL** database, a **Qdrant** vector database for the knowledge base, and a **Redis** cache.

## Tech Stack

  * **Backend:** Python, FastAPI, LangChain, DSPy, Qdrant Client
  * **Frontend:** TypeScript, React, Next.js, Tailwind CSS, React Query
  * **LLM:** Groq (easily configurable for other providers like Gemini)
  * **Web Search:** Tavily API
  * **Database:** Qdrant (VectorDB), PostgreSQL
  * **Deployment:** Docker

## Setup and Installation

Follow these steps to get the Math Routing Agent running locally.

#### 1\. Prerequisites

  * Docker and Docker Compose
  * Python 3.11+
  * Node.js and npm

#### 2\. Clone the Repository

```bash
git clone <your-repo-url>
cd math-routing-agent
```

#### 3\. Configure Environment Variables

Create a `.env` file in the project's root directory by copying the example file:

```bash
cp .env.example .env
```

Now, open the `.env` file and add your API keys for **Groq** and **Tavily**.

#### 4\. Launch Infrastructure with Docker

This command will start the Qdrant, PostgreSQL, and Redis containers.

```bash
docker-compose up -d
```

#### 5\. Set Up the Backend

```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # On Windows (Git Bash)
# source venv/bin/activate    # On macOS/Linux
pip install -r requirements.txt
```

#### 6\. Set Up the Frontend

```bash
cd ../frontend
npm install
```

## Running the Application

1.  **Start the Backend Server:**
    In the `backend` directory, run:

    ```bash
    uvicorn app.main:app --reload
    ```

    The backend will be available at `http://127.0.0.1:8000`.

2.  **Start the Frontend Server:**
    In a new terminal, from the `frontend` directory, run:

    ```bash
    npm run dev
    ```

    The frontend will be available at `http://localhost:3000`.

You can now open your browser to `http://localhost:3000` to start using the Math Routing Agent.
