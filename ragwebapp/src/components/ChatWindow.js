// src/components/ChatWindow.js
import { useEffect, useRef } from 'react';

// reusable
const SuggestionChip = ({ query, onClick }) => (
    <button className="suggestion-chip" onClick={() => onClick(query)}>{query}</button>
);

const Message = ({ sender, text }) => {
    const isUser = sender === 'user';
    
    
    const formatText = (rawText) => {
        return rawText.split('\n').map((line, index) => (
            <span key={index}>{line}<br/></span>
        ));
    };
    
    return (
        <div className={`message ${isUser ? 'user-message' : 'bot-message'}`}>
            <div className="message-avatar">
                {isUser ? 
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" /></svg> :
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M19.96 11.45c.03-.15.04-.3.04-.45 0-2.76-2.24-5-5-5s-5 2.24-5 5c0 .15.01.3.04.45C7.17 12.23 5 14.86 5 18v2h14v-2c0-3.14-2.17-5.77-5.04-6.55zM8 12c0-1.66 1.34-3 3-3s3 1.34 3 3-1.34 3-3 3-3-1.34-3-3z" /></svg>
                }
            </div>
                <span className="message-sender">{isUser ? 'You' : 'FinSecureAI'}</span>
            <div className="message-content-wrapper">
                <div className="message-bubble">{formatText(text)}</div>
            </div>
        </div>
    );
};

const TypingIndicator = () => (
    <div className="message bot-message typing-indicator">
         <div className="message-avatar">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M19.96 11.45c.03-.15.04-.3.04-.45 0-2.76-2.24-5-5-5s-5 2.24-5 5c0 .15.01.3.04.45C7.17 12.23 5 14.86 5 18v2h14v-2c0-3.14-2.17-5.77-5.04-6.55zM8 12c0-1.66 1.34-3 3-3s3 1.34 3 3-1.34 3-3 3-3-1.34-3-3z" /></svg>
        </div>
        <div className="message-content-wrapper">
            <span className="message-sender">FinSecureAI</span>
            <div className="message-bubble">
                <span className="dot"></span><span className="dot"></span><span className="dot"></span>
            </div>
        </div>
    </div>
);

//main ChatWindow 
const ChatWindow = ({ messages, isLoading, onSuggestionClick }) => {
    const chatEndRef = useRef(null);

    // Auto-scroll to the bottom when messages change
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <main id="chatScreen" className="chat-content-area">
            <div id="messageList" className="message-list">
                {/* Show initial prompt only if there are no messages */}
                {messages.length === 0 && !isLoading && (
                    <div id="initialPrompt" className="initial-prompt-container">
                        <h2>Ask me anything financial.</h2>
                        <p className="sub-prompt">I can help with general questions about savings, investments, budgeting, and more.</p>
                        <div id="suggestionChips" className="suggestion-chips">
                            <SuggestionChip query="Who is the CEO of Apple?" onClick={onSuggestionClick} />
                            <SuggestionChip query="Show me the latest 10-K from Apple." onClick={onSuggestionClick} />
                            <SuggestionChip query="What does Tesla say about autonomous driving, also give me strategic recommendations." onClick={onSuggestionClick} />
                        </div>
                    </div>
                )}
                
                {/* Render the list of messages from state */}
                {messages.map((msg, index) => (
                    <Message key={index} sender={msg.sender} text={msg.text} />
                ))}

                {/* Show typing indicator if loading */}
                {isLoading && <TypingIndicator />}

                {/* This empty div is a reference point for auto-scrolling */}
                <div ref={chatEndRef} />
            </div>
        </main>
    );
};

export default ChatWindow;