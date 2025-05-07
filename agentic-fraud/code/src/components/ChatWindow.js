import React, { useState, useEffect, useRef } from 'react';
import './ChatWindow.css';
import Message from './Message';
import Anthropic from '@anthropic-ai/sdk';

const ChatWindow = () => {
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm Claude. How can I help you today?", sender: 'assistant' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const [conversationId, setConversationId] = useState(null);
  
  // Reference to store all messages for context to Claude
  const conversationHistory = useRef([
    { role: "assistant", content: "Hello! I'm Claude. How can I help you today?" }
  ]);

  // Initialize Anthropic client using environment variable
  // In production, API keys should be handled by a backend service
  const API_KEY = process.env.REACT_APP_ANTHROPIC_API_KEY; 
  
  // Check if API key is available
  const [apiKeyMissing, setApiKeyMissing] = useState(!API_KEY);
  
  // Create Anthropic client if API key is available
  const anthropic = API_KEY ? new Anthropic({
    apiKey: API_KEY,
  }) : null;

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    // Add user message to UI
    const userMessage = {
      id: messages.length + 1,
      text: input,
      sender: 'user'
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    // Add to conversation history
    conversationHistory.current.push({ role: "user", content: input });
    
    setInput('');
    setIsLoading(true);
    
    // Check if API key is missing
    if (apiKeyMissing) {
      const errorMessage = {
        id: messages.length + 2,
        text: "Error: Anthropic API key is missing. Please add your API key to the .env file as described in README.md.",
        sender: 'assistant'
      };
      
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
      return;
    }
    
    try {
      // For development/testing - use local fallback if API key issues persist
      let response;
      
      // Add debug statement for API key
      console.log("API key first 4 chars:", API_KEY ? API_KEY.substring(0, 4) + "..." : "missing");
      
      // Local fallback mode for testing
      const useFallbackMode = !API_KEY || API_KEY === "YOUR_ANTHROPIC_API_KEY";
      
      if (useFallbackMode) {
        console.log("Using fallback mode (no API call)");
        // Simulate Claude response for testing without API
        response = "This is a fallback response since no valid API key is available. For the full experience, please add your Anthropic API key to the .env file.";
      } else {
        // Call Claude API with conversation history
        response = await sendMessageToClaude(input);
      }
      
      // Add Claude's response to UI
      const assistantMessage = {
        id: messages.length + 2,
        text: response,
        sender: 'assistant'
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
      // Add to conversation history
      conversationHistory.current.push({ role: "assistant", content: response });
      
    } catch (error) {
      console.error('Error sending message to Claude:', error);
      
      let errorMessage = "Sorry, I encountered an error processing your request. Please try again later.";
      
      // Extract more detailed error information
      if (error.status) {
        console.log(`Status code: ${error.status}`);
      }
      
      // Provide more specific error messages for common issues
      if (error.status === 401) {
        errorMessage = "Authentication error: Please check that your Anthropic API key is valid.";
      } else if (error.status === 429) {
        errorMessage = "Rate limit exceeded: The application has made too many requests to the Claude API.";
      } else if (error.message) {
        // Include the actual error message for debugging
        errorMessage = `Error: ${error.message}`;
      }
      
      // Add error message to UI
      setMessages(prev => [...prev, {
        id: messages.length + 2,
        text: errorMessage,
        sender: 'assistant'
      }]);
      
      // Add error to conversation history
      conversationHistory.current.push({ 
        role: "assistant", 
        content: errorMessage 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessageToClaude = async (userInput) => {
    // In a real production app, this should be a call to your backend server
    // which would then securely call the Anthropic API
    // This direct frontend implementation is for demonstration purposes only
    try {
      // Add debugging logs
      console.log("Sending message to Claude. API Key exists:", !!API_KEY);
      console.log("Conversation history:", JSON.stringify(conversationHistory.current));
      
      // System prompt for instructing Claude's behavior
      // For the workshop, this can be modified to demonstrate different safety measures
      const systemPrompt = 
        "You are Claude, an AI assistant by Anthropic. " +
        "This is a safety workshop about agentic fraud. " +
        "Your responses should be helpful, harmless, and honest. " +
        "Keep your responses concise and focused on the user's questions. " +
        "For demonstration purposes, explain AI safety concepts when relevant. " +
        "Never assist with illegal activities or harmful plans.";
      
      // Simplified API call to reduce potential issues
      const response = await anthropic.messages.create({
        model: "claude-3-7-sonnet-20250201", // Using Claude 3.7 Sonnet
        max_tokens: 1000,
        messages: conversationHistory.current,
        system: systemPrompt
      });
      
      // Debug response
      console.log("Claude API response:", response);
      
      // Store conversation ID if API returns one and we don't have one yet
      if (response.id && !conversationId) {
        setConversationId(response.id);
      }
      
      // Safely access response content
      if (response && response.content && response.content.length > 0) {
        return response.content[0].text;
      } else {
        throw new Error("Unexpected response format from Claude API");
      }
    } catch (error) {
      console.error("Error calling Claude API:", error);
      console.error("Error details:", JSON.stringify(error, null, 2));
      throw error;
    }
  };

  return (
    <div className="chat-window">
      {apiKeyMissing && (
        <div className="api-key-warning">
          <strong>API Key Missing:</strong> Please add your Anthropic API key to continue.
          <p>Follow the instructions in the README.md file to set up your .env file.</p>
        </div>
      )}
      
      <div className="chat-messages">
        {messages.map(message => (
          <Message key={message.id} message={message} />
        ))}
        {isLoading && (
          <div className="message assistant">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          placeholder="Type your message here..."
          disabled={isLoading}
          className="chat-input"
        />
        <button type="submit" disabled={isLoading || !input.trim()} className="send-button">
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatWindow;