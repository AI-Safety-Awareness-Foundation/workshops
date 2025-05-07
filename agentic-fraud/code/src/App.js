import React, { useState } from 'react';
import './App.css';
import ChatWindow from './components/ChatWindow';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Safety Workshop: Agentic Fraud Demo</h1>
      </header>
      <main className="app-main">
        <div className="content-warning">
          <strong>Educational Purpose Only:</strong> This application demonstrates potential 
          AI fraud risks for educational and research purposes.
        </div>
        <ChatWindow />
      </main>
      <footer className="app-footer">
        <p>AI Safety Research Workshop</p>
      </footer>
    </div>
  );
}

export default App;