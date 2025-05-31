# Google Calendar Assistant

This Python application creates a server that communicates with Google's Gemini AI to interact with your Google Calendar. It uses the Model Context Protocol (MCP) to connect with Google Calendar via SSE (Server-Sent Events).

## How It Works

The application follows this flow:

1. **Server Initialization**:
   - Connects to the MCP server that's already connected to your Google account
   - Retrieves the list of available tools to interact with Google Calendar

2. **User Request Processing**:
   - User makes a request (e.g., "Tell me what are my meetings tomorrow")
   - The request is sent to Gemini AI along with the available tools list

3. **AI Processing**:
   - Gemini AI analyzes the request to determine if tools are needed
   - If tools are needed (e.g., for calendar operations), Gemini will use the MCP server to interact with Google Calendar
   - If no tools are needed (e.g., casual conversation), Gemini will respond directly

4. **Response Delivery**:
   - Gemini sends its analysis/response back to the user
   - The cycle continues with new user requests

## Prerequisites

- Python 3.8 or higher
- Gemini API key
- Access to the MCP server connected to your Google Calendar

## Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Make sure your `.env` file contains your Gemini API key and MCP server URL:

```
GEMINI_API_KEY=your_api_key_here
MCP_SERVER_URL=your_mcp_server_url_here
```

## Usage

Run the application with:

```bash
python calendar_assistant.py
```

In the interactive chat:
- Type your questions or commands about your calendar
- Type `refresh` to clear the conversation history
- Type `quit` to exit the application

## Example Interactions

Here are some examples of how you can interact with the assistant:

- "Tell me what are my meetings tomorrow"
- "Create a meeting for tomorrow at 5PM with the name 'bank stuff'"
- "What's my schedule for next week?"
- "Cancel my 3PM meeting today"
- "Reschedule my meeting with John to Friday at 10AM"

## Successful Tests

The following features have been successfully tested:

- "Show me events for next week/tomorrow/next month"
- "Create an event tomorrow at 3 PM"
- "Delete the event called test tomorrow"

The assistant correctly handles relative dates and local time zones for creating and searching events.

## Code Structure

- `calendar_assistant.py`: Main application file containing the CalendarAssistant class and chat loop
- `requirements.txt`: List of required Python packages
- `.env`: Environment variables file for storing the Gemini API key
- `logs/`: Directory for storing application logs

## Technical Details

- **MCP (Model Context Protocol)**: An open standard for connecting AI applications with external tools and data
- **Gemini 2.0 Flash**: Google's AI model used for processing user requests
- **SSE (Server-Sent Events)**: Protocol used for communication with the MCP server

## Troubleshooting

If you encounter issues:
1. Check the logs in the `logs/` directory
2. Ensure your Gemini API key is correct
3. Verify that the MCP server URL is accessible

## Future Improvements

The following improvements are planned for future versions:

1. **Enhanced User Interface**:
   - Remove logs from the interface for a cleaner user experience
   - Implement a web interface for easier access

2. **New Features**:
   - Add voice control for more natural interaction

3. **Technical Improvements**:
   - Optimize MCP connection management for better stability
   - Improve error handling and user messages
   - Add automated tests to ensure reliability
