// File: src/app/api/ask/route.js

import { NextResponse } from 'next/server';

export async function POST(request) {
  // get user query
  const { query } = await request.json();

  // get backend API URL from env file
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  // check if api url is empty
  if (!apiBaseUrl) {
    return NextResponse.json(
      { message: 'Backend API URL is not configured.' },
      { status: 500 }
    );
  }

  try {
    // send the user query to the backend API
    const response = await fetch(`${apiBaseUrl}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      // check for error response from backend
      const errorData = await response.text();
      console.error('Backend API Error:', errorData);
      return NextResponse.json(
        { message: `Backend API failed with status: ${response.status}` },
        { status: response.status }
      );
    }

    // get the answer from the llm backend
    const data = await response.json();

    // send the response to be displayed in the chat window
    return NextResponse.json(data);

  } catch (error) {
    console.error('Error communicating with the Colab backend:', error);
    return NextResponse.json(
      { message: 'An internal server error occurred.' },
      { status: 500 }
    );
  }
}