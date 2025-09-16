# 🧠 Math Routing Agent

The Math Routing Agent is a sophisticated **Agentic-RAG (Retrieval-Augmented Generation)** system designed to function as an AI-powered mathematical professor. It understands mathematical questions and provides detailed, step-by-step solutions tailored for students. The system leverages a dynamic knowledge base, web search capabilities, and a human-in-the-loop feedback mechanism to continuously improve its accuracy and clarity.

This project fulfills all core requirements of the Generative AI assignment, including the bonus for using the **DSPy library** for feedback-driven optimization.

## ✨ Core Features

  * **Agentic RAG Pipeline:** The system first queries a **Qdrant vector database** for existing solutions. If no relevant information is found, it automatically performs a web search to gather context before generating an answer.
  * **🛡️ AI Gateway & Guardrails:** Implements robust input and output guardrails to ensure all content is strictly mathematical and educational, filtering out irrelevant or inappropriate queries.
  * **🧑‍🏫 Human-in-the-Loop Feedback:** Users can provide feedback on the quality of solutions. This feedback is processed by a dedicated feedback agent to identify areas for improvement.
  * **🧠 DSPy-Powered Self-Learning (Bonus):** Utilizes the **DSPy library** to analyze aggregated user feedback and automatically fine-tune the language model, allowing the agent to learn and refine its responses over time.
  * **🌐 MCP for Web Search:** Uses the Model Context Protocol (MCP) to interact with a dedicated web search server, ensuring a modular and scalable architecture for tool use.

## 🏗️ System Architecture

The application follows a modern, decoupled architecture designed for scalability and maintainability.

  * **Frontend:** A responsive and interactive user interface built with **React (Next.js)**. It communicates with the backend via RESTful APIs and WebSockets for real-time updates.
  * **Backend:** A powerful **FastAPI** server that orchestrates the entire agentic workflow. It houses the various agents (`MathAgent`, `RoutingAgent`, `GuardrailsAgent`, `FeedbackAgent`) and services that form the core logic of the application.
  * **Infrastructure (Docker Compose):** A **Docker Compose** setup manages all the necessary backing services, ensuring a consistent and isolated development environment. This includes:
      * **Qdrant:** The vector database that serves as the dynamic knowledge base for the RAG system.
      * **PostgreSQL:** A relational database for storing structured data (as needed).
      * **Redis:** An in-memory data store for caching and other high-speed operations.

## ⚙️ Agentic RAG Pipeline Flow

The system follows a sophisticated, multi-step process to handle each user query, ensuring accuracy and relevance.

1.  **Input & Guardrails:** A user submits a question through the React frontend. The request is sent to the FastAPI backend, where the `GuardrailsAgent` first validates that the question is mathematical and appropriate.
2.  **Routing Decision:** The `RoutingAgent` takes over and first queries the **Qdrant knowledge base** to see if a similar question has been answered before.
3.  **Knowledge Base Retrieval:** If a highly similar solution is found, it is retrieved and used as the primary context for generating the answer.
4.  **Web Search (MCP):** If the knowledge base does not contain a relevant answer, the `RoutingAgent` triggers the `MCPWebSearchService`. This service calls the `search_server.py` script to perform an external web search using the Tavily API.
5.  **Solution Generation:** The retrieved information (either from the knowledge base or web search) is passed to the **Groq LLM** to generate a clear, step-by-step solution.
6.  **Knowledge Base Update:** If the solution was generated from a web search and has a high confidence score, it is converted into an embedding and **stored in the Qdrant database**, allowing the agent to learn from new problems.
7.  **Human-in-the-Loop:** The final solution is presented to the user, who can provide feedback. This feedback is sent to the `FeedbackAgent` and used by the `DSPyFeedbackOptimizer` to refine the agent's performance for future questions, thus completing the loop.

## 🚀 Getting Started

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

This command will start the Qdrant, PostgreSQL, and Redis containers in the background.

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

## 🏃 Running the Application

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
