// src/components/InputArea.js
import { useRef, useEffect } from 'react';

const InputArea = ({ inputValue, onInputChange, onSendMessage, isLoading }) => {
    const textareaRef = useRef(null);

    // Auto-resize the textarea height based on content
    useEffect(() => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            const newHeight = textarea.scrollHeight;
            textarea.style.height = `${newHeight}px`;
        }
    }, [inputValue]);

    const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          onSendMessage(event);
        }
    };

    return (
        <div className="input-area-wrapper">
            <form id="chatForm" className="input-form" onSubmit={onSendMessage}>
                <textarea 
                    ref={textareaRef}
                    id="userInput" 
                    placeholder="Type your financial question..." 
                    rows="1"
                    value={inputValue}
                    onChange={onInputChange}
                    onKeyPress={handleKeyPress}
                    disabled={isLoading}
                />
                <button id="sendButton" type="submit" aria-label="Send message" disabled={isLoading || !inputValue.trim()}>
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                        <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                    </svg>
                </button>
            </form>
        </div>
    );
};

export default InputArea;