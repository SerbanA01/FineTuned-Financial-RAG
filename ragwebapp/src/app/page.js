// src/app/page.js
'use client';

import { useState } from 'react';
import ChatWindow from '@/components/ChatWindow';
import InputArea from '@/components/InputArea';

export default function Home() {

  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (event) => {
    //prevent default behavior
    event.preventDefault();

    const userMessageText = inputValue.trim();
    if (!userMessageText) return; 

    // show the user's message immediately
    const newUserMessage = { sender: 'user', text: userMessageText };
    setMessages(prevMessages => [...prevMessages, newUserMessage]);

    // clear input and start the loading animation
    setInputValue('');
    setIsLoading(true);

    try {
      // send the user query to the backend API
      const response = await fetch("/api/ask", { 
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessageText }) 
      });
    
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
    
      const data = await response.json();
      const botResponse = data.answer;

      // add the bot's response to the chat
      const newBotMessage = { sender: 'bot', text: botResponse };
      setMessages(prevMessages => [...prevMessages, newBotMessage]);

    } catch (error) {
      console.error("Failed to send message:", error);
      // if there's an error, show a generic error message in the chat
      const errorMessage = { sender: 'bot', text: "❌ Sorry, I couldn't connect to the server. Please try again later." };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      // stop the loading animation
      setIsLoading(false);
    }
  };

  // This function will handle input changes
  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };
  

  const handleSuggestionClick = (query) => {
    setInputValue(query);
  };

  return (
    <main>
      <div className="app-container">
        <header className="main-header">
          <h1 className="header-title centered">FinSecure<span className="accent">AI</span></h1>
        </header>
        
        <ChatWindow 
          messages={messages} 
          isLoading={isLoading}
          onSuggestionClick={handleSuggestionClick} 
        />
        
        <InputArea 
          inputValue={inputValue}
          onInputChange={handleInputChange}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </div>
    </main>
  );
}