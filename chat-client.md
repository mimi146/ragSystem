# Chat Client Implementation Plan

## Objective
Create a clean, modern, and reader-friendly web-based chat interface for the Weather RAG API. The client will support real-time streaming responses and maintain a history of chat sessions in a left-hand sidebar.

## Architecture & Tech Stack
- **Frontend:** Vanilla HTML5, CSS3, and JavaScript. No heavy build steps (React/Vue) required.
- **Serving:** Served directly from the FastAPI backend using `fastapi.staticfiles.StaticFiles`.
- **Markdown Rendering:** Use `marked.js` (via CDN) to ensure the LLM's text output (which often includes markdown for bolding, lists, or code) is beautifully formatted and highly readable.
- **Styling:** Custom CSS focusing on a clean aesthetic (similar to modern chat apps), using Flexbox/Grid for a responsive layout.

## UI/UX Design
The interface will be divided into two main sections:
1. **Left Sidebar (History & Management):**
   - A list of past conversation threads (stored locally in the browser's `localStorage` or managed in-memory per session).
   - A button to start a "New Chat".
   - A section or button to upload new documents (PDF/DOCX) to the RAG knowledge base.
2. **Main Chat Area (Right Section):**
   - A scrollable view of the current conversation.
   - Distinct styling for User messages (right-aligned, solid background) and Assistant messages (left-aligned, light background).
   - **Real-time Streaming:** The Assistant's message bubble will update dynamically as tokens stream in from the `/weather/ask-stream` endpoint.
   - **Reader-Friendly Text:** Ample line height, readable font stack (e.g., Inter, Roboto, or system fonts), and markdown parsing for structured responses.
   - An input field pinned to the bottom with a send button.

## Implementation Steps

### Phase 1: Backend Updates (`main.py`)
- Import `StaticFiles` from `fastapi.staticfiles`.
- Create a `static` directory in the project root.
- Mount the static directory to serve the frontend (e.g., `app.mount("/chat", StaticFiles(directory="static", html=True), name="chat")`).

### Phase 2: Frontend Structure (`static/index.html`)
- Create the HTML skeleton with a sidebar `<aside>` and a main `<main>` area.
- Include CDNs for icons (e.g., FontAwesome) and markdown parsing (`marked.js`).

### Phase 3: Styling (`static/style.css`)
- Define CSS variables for a cohesive color palette (backgrounds, text, borders, accents).
- Implement a CSS Grid/Flexbox layout to position the sidebar and chat area.
- Style the chat bubbles for maximum readability (padding, border-radius, line-height).
- Style the document upload input and buttons.

### Phase 4: Interaction Logic (`static/script.js`)
- **State Management:** Keep track of the current `user_id` (e.g., generating a random UUID on load) and the active conversation.
- **Upload Handling:** Wire up the file input to POST to `/weather/upload-document` and display upload progress/success.
- **Streaming Chat:** 
  - Capture user input and render it immediately.
  - Create a placeholder bubble for the Assistant.
  - Use the native `fetch` API to POST to `/weather/ask-stream`.
  - Read the `ReadableStream` from the response body.
  - Append tokens to the Assistant's bubble in real-time, passing the accumulated text through `marked.parse()` to render markdown.
- **History Management:** Save chat history to `localStorage` and render the list in the left sidebar, allowing the user to switch between past conversations.

## Verification & Testing
1. Ensure the static files load correctly at `http://localhost:8000/chat`.
2. Test uploading a document and verifying the success message.
3. Test querying the LLM and verify that the text streams smoothly into the UI and renders markdown correctly.
4. Verify that creating a new chat clears the main area and adds an entry to the sidebar history.