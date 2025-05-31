"""
Google Calendar Assistant using MCP and Gemini

This script creates a server that communicates with Google's Gemini AI to interact with a Google Calendar.
It uses the Model Context Protocol (MCP) to connect with Google Calendar via SSE.

Flow:
1. Server initialization - Connect to MCP server and get available tools
2. User request - Process user's query
3. AI processing - Send to Gemini with tools list
4. Tool usage - If needed, use MCP to interact with Google Calendar
5. Response - Return AI's response to the user
"""

import os
import asyncio
import logging
import json
from typing import List, Dict, Any, Optional

# Import MCP client libraries
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.shared.exceptions import McpError

# Import Gemini API
import google.generativeai as genai
from google.generativeai import types as genai_types

# Import dotenv for environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/calendar_assistant.log"),
        logging.StreamHandler()  # Keep console output as well
    ]
)
logger = logging.getLogger(__name__)

# Define the system prompt for Gemini
SYSTEM_PROMPT = """You are an AI assistant specialized in managing Google Calendar. You have access to tools that allow you to interact directly with the user's calendar.

Your capabilities include:
1. Viewing calendar events (appointments, meetings, etc.)
2. Creating new events in the calendar
3. Modifying existing events
4. Deleting events

When the user asks you a question about their calendar or requests an action related to their calendar, ALWAYS use the appropriate tools to respond to their request. Never invent answers about the calendar without using the tools.

IMPORTANT: Here are the EXACT parameters for each tool:

- For GOOGLECALENDAR_FIND_EVENT:
  * "calendar_id": "primary" (Optional, defaults to primary)
  * "query": "Meeting with John" (Optional, search term to find events)
  * "timeMin": "2025-05-31T00:00:00" (Optional, lower bound for event's end time)
  * "timeMax": "2025-06-07T23:59:59" (Optional, upper bound for event's start time)
  * "single_events": true (Optional, defaults to true)
  * "max_results": 10 (Optional, defaults to 10, max 2500)
  * "order_by": "startTime" (Optional, can be "startTime" or "updated")
  * "show_deleted": false (Optional)

- For GOOGLECALENDAR_CREATE_EVENT:
  * "summary": "Meeting title" (Optional but recommended)
  * "start_datetime": "2025-05-31T15:00:00" (REQUIRED, format YYYY-MM-DDTHH:MM:SS)
  * "timezone": "<use local timezone" (IANA timezone name)
  * "event_duration_hour": 1 (Optional, defaults to 0)
  * "event_duration_minutes": 30 (Optional, defaults to 30)
  * "description": "Meeting details" (Optional)
  * "location": "Conference Room A" (Optional)
  * "attendees": ["email1@example.com", "email2@example.com"] (Optional)
  * "create_meeting_room": true (Optional, adds Google Meet link)
  * "calendar_id": "primary" (Optional, defaults to primary)

- For GOOGLECALENDAR_DELETE_EVENT:
  * "event_id": "event123" (REQUIRED)
  * "calendar_id": "primary" (Optional, defaults to primary)

When the user mentions a date like "tomorrow" or "next Friday", you MUST convert it to an actual date based on the current date provided in the user's message.

For example:
- If today is May 31, 2025, and user says "what are my meetings this week", use timeMin="2025-05-31T00:00:00" and timeMax="2025-06-07T23:59:59"
- If today is May 31, 2025, and user says "create a meeting tomorrow at 3pm", use start_datetime="2025-06-01T15:00:00". Modify this accordingly with the user timezone.

For general questions not related to the calendar, respond normally as an AI assistant.

Examples of queries and expected actions:
- "What are my appointments tomorrow?" → Use GOOGLECALENDAR_FIND_EVENT with {"timeMin": "2025-06-01T00:00:00", "timeMax": "2025-06-01T23:59:59"}
- "Create a meeting on Thursday at 3pm" → Use GOOGLECALENDAR_CREATE_EVENT with {"summary": "Meeting", "start_datetime": "2025-06-05T15:00:00", "event_duration_hour": 1, "event_duration_minutes": 0} Change this according to the user timezone
- "Cancel my 2pm appointment" → First use GOOGLECALENDAR_FIND_EVENT to find the event_id, then use GOOGLECALENDAR_DELETE_EVENT with {"event_id": "found_event_id"}
- "Quels sont mes rendez-vous cette semaine?" → Use GOOGLECALENDAR_FIND_EVENT with {"timeMin": "2025-05-31T00:00:00", "timeMax": "2025-06-07T23:59:59"}

Feel free to use the available tools for any calendar-related question, regardless of the language the user is using (English, French, etc.).
Take into account the communicated timezone when creating events.
Respect the camelCase naming convention for the tools.
"""

class CalendarAssistant:
    """
    A class that handles the interaction between the user, Gemini AI, and Google Calendar.
    
    This class connects to an MCP server that provides tools for interacting with Google Calendar,
    and uses Gemini AI to process user requests and determine when to use those tools.
    """
    
    def __init__(self, mcp_server_url: str, gemini_api_key: str):
        """
        Initialize the Calendar Assistant.
        
        Args:
            mcp_server_url: URL of the MCP server that's connected to Google Calendar
            gemini_api_key: API key for Gemini AI
        """
        self.mcp_server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self.available_tools = []
        
        # Initialize Gemini client
        genai.configure(api_key=gemini_api_key)
        self.model = "gemini-2.0-flash"  # Using the flash model as specified
        
        logger.info(f"Calendar Assistant initialized with MCP server: {mcp_server_url}")
    
    async def connect_to_mcp_server(self):
        """
        Connect to the MCP server and retrieve available tools.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create SSE client and connect to the server
            self._streams_context = sse_client(url=self.mcp_server_url)
            streams = await self._streams_context.__aenter__()
            self._session_context = ClientSession(*streams)
            self.session = await self._session_context.__aenter__()
            await self.session.initialize()
            response = await self.session.list_tools()
            self.available_tools = response.tools
            
            # Créer un fichier log pour les paramètres des outils
            with open('tools_parameters.log', 'w') as f:
                f.write("===== OUTILS DISPONIBLES ET LEURS PARAMÈTRES =====\n\n")
                
                for tool in self.available_tools:
                    f.write(f"\nOutil: {tool.name}\n")
                    if hasattr(tool, 'description') and tool.description:
                        f.write(f"Description: {tool.description}\n")
                    
                    # Extraire et afficher les paramètres attendus
                    if hasattr(tool, 'parameters'):
                        f.write("Paramètres attendus:\n")
                        try:
                            # Essayer de parser les paramètres s'ils sont au format JSON
                            if isinstance(tool.parameters, str):
                                params = json.loads(tool.parameters)
                                for param_name, param_info in params.get('properties', {}).items():
                                    required = 'Requis' if param_name in params.get('required', []) else 'Optionnel'
                                    param_type = param_info.get('type', 'inconnu')
                                    param_desc = param_info.get('description', 'Pas de description')
                                    f.write(f"  - {param_name} ({param_type}): {param_desc} [{required}]\n")
                            else:
                                f.write(f"  Format non-JSON: {tool.parameters}\n")
                        except Exception as e:
                            f.write(f"  Impossible de parser les paramètres: {e}\n")
                            f.write(f"  Données brutes: {tool.parameters}\n")
                
                f.write("\n================================================\n")
            
            print("\nInformations sur les outils enregistrées dans le fichier 'tools_parameters.log'\n")
            
            tool_names = [tool.name for tool in self.available_tools]
            logger.info(f"Connected to MCP server. Available tools: {tool_names}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}", exc_info=True)
            return False
    
    def _convert_tools_to_gemini_format(self) -> List[Dict[str, Any]]:
        """
        Convert MCP tools to Gemini's function declaration format.
        
        Returns:
            List of function declarations in Gemini's format
        """
        # Map JSON schema types to Gemini types
        type_mapping = {
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
            "string": "STRING"
        }
        
        gemini_tools = []
        for tool in self.available_tools:
            # Create basic tool structure
            function_declaration = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {"type": "OBJECT", "properties": {}, "required": []}
            }
            
            # Convert schema if available
            if tool.inputSchema:
                schema = tool.inputSchema
                
                # Add properties from the schema
                if hasattr(schema, "properties"):
                    for prop_name, prop_details in schema.properties.items():
                        prop_type = getattr(prop_details, "type", "STRING")
                        prop_type = type_mapping.get(prop_type.lower(), "STRING")
                            
                        property_schema = {"type": prop_type}
                        if hasattr(prop_details, "description"):
                            property_schema["description"] = prop_details.description
                            
                        function_declaration["parameters"]["properties"][prop_name] = property_schema
                        
                # Add required properties
                if hasattr(schema, "required"):
                    function_declaration["parameters"]["required"] = schema.required
                    
            gemini_tools.append(function_declaration)
        
        return gemini_tools
    
    def _prepare_gemini_chat_history(self, previous_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare chat history in Gemini's format.
        
        Args:
            previous_messages: List of previous messages in the conversation
            
        Returns:
            Chat history formatted for Gemini
        """
        chat_history = []
        if not previous_messages:
            return chat_history
            
        for message in previous_messages:
            if message["role"] == "user" and isinstance(message["content"], str):
                chat_history.append({
                    "role": "user",
                    "parts": [{"text": message["content"]}]
                })
            elif message["role"] == "assistant" and isinstance(message["content"], str):
                chat_history.append({
                    "role": "model",
                    "parts": [{"text": message["content"]}]
                })
        
        return chat_history
    
    async def _process_gemini_response(self, response, final_text, messages):
        """
        Process the response from Gemini, including any function calls.
        
        Args:
            response: The response from Gemini
            final_text: List to accumulate the final text response
            messages: List to accumulate the conversation history
            
        Returns:
            Tuple of (final_text, messages)
        """
        if not hasattr(response, "candidates") or not response.candidates:
            logger.warning("No candidates in Gemini response")
            final_text.append("I couldn't generate a proper response.")
            return final_text, messages
            
        candidate = response.candidates[0]
        if not hasattr(candidate, "content") or not hasattr(candidate.content, "parts"):
            logger.warning("No content or parts in Gemini response")
            final_text.append("I received an incomplete response.")
            return final_text, messages
            
        # Process text and function calls
        for part in candidate.content.parts:
            # Process text part
            if hasattr(part, "text") and part.text:
                final_text.append(part.text)
                
            # Process function call part
            if hasattr(part, "function_call") and part.function_call:
                function_call = part.function_call
                tool_name = function_call.name
                
                # Parse tool arguments
                tool_args = {}
                try:
                    if hasattr(function_call.args, "items"):
                        for k, v in function_call.args.items():
                            tool_args[k] = v
                    else:
                        # Fallback if it's a string
                        args_str = str(function_call.args)
                        if args_str.strip():
                            tool_args = json.loads(args_str)
                except Exception as e:
                    logger.error(f"Failed to parse function args: {e}", exc_info=True)
                
                # Log and print the tool call for the user to see
                tool_call_message = f"Gemini is calling tool: {tool_name} with parameters: {json.dumps(tool_args, indent=2)}"
                logger.info(tool_call_message)
                print(f"\n{tool_call_message}\n")
                
                # Execute tool call
                try:
                    logger.debug(f"Calling tool {tool_name} with args {tool_args}")
                    try:
                        # Try to call the tool
                        result = await self.session.call_tool(tool_name, tool_args)
                    except McpError as mcp_error:
                        # Handle connection closed error
                        if "Connection closed" in str(mcp_error):
                            logger.warning("MCP connection closed. Attempting to reconnect...")
                            print("\nMCP connection closed. Attempting to reconnect...\n")
                            
                            # Reconnect to the MCP server
                            await self.cleanup()
                            connected = await self.connect_to_mcp_server()
                            
                            if connected:
                                logger.info("Successfully reconnected to MCP server. Retrying tool call...")
                                print("\nSuccessfully reconnected to MCP server. Retrying tool call...\n")
                                # Try the tool call again
                                result = await self.session.call_tool(tool_name, tool_args)
                            else:
                                raise Exception("Failed to reconnect to MCP server")
                        else:
                            # Re-raise other MCP errors
                            raise
                    
                    # Format the result content
                    result_content = result.content if hasattr(result, "content") else str(result)
                    
                    # Create a new chat for follow-up
                    # First, get the tools again
                    gemini_tools = self._convert_tools_to_gemini_format()
                    gemini_tool_config = genai_types.Tool(function_declarations=gemini_tools)
                    
                    follow_up_chat = genai.GenerativeModel(
                        model_name=self.model,
                        generation_config={"temperature": 0.2},
                        tools=[gemini_tool_config],
                        system_instruction=SYSTEM_PROMPT
                    ).start_chat()
                    
                    # Send the function call result to the model
                    logger.debug(f"Tool {tool_name} returned: {result_content}")
                    
                    # Create a message to send to the model with the tool results
                    follow_up_message = f"I used the {tool_name} tool with these parameters: {json.dumps(tool_args)}. Here are the results: {result_content}. Please provide a helpful response based on these results."
                    
                    # Send function response to get final answer
                    follow_up_response = follow_up_chat.send_message(follow_up_message)
                    
                    # Extract text from follow-up response
                    if hasattr(follow_up_response, "candidates") and follow_up_response.candidates:
                        follow_up_candidate = follow_up_response.candidates[0]
                        if (hasattr(follow_up_candidate, "content") and 
                            hasattr(follow_up_candidate.content, "parts")):
                            
                            follow_up_text = ""
                            for follow_up_part in follow_up_candidate.content.parts:
                                if hasattr(follow_up_part, "text"):
                                    follow_up_text += follow_up_part.text
                                    
                            if follow_up_text:
                                final_text.append(follow_up_text)
                                messages.append({
                                    "role": "assistant", 
                                    "content": follow_up_text
                                })
                            else:
                                final_text.append("I received the tool results but couldn't generate a follow-up response.")
                        
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    final_text.append(error_msg)
                
        return final_text, messages
    
    async def process_query(self, query: str, previous_messages: Optional[List[Dict[str, Any]]] = None):
        """
        Process a user query using Gemini and the MCP tools.
        
        Args:
            query: The user's query
            previous_messages: Previous conversation history
            
        Returns:
            Tuple of (response_text, updated_messages)
        """
        if not self.session:
            logger.error("Session not initialized. Please connect to MCP server first.")
            return "I'm not connected to the calendar service. Please try again later.", []
        
        # Add current date and time to the query
        from datetime import datetime
        now = datetime.now()
        local_timezone = now.astimezone().tzinfo
        formatted_date = now.strftime("%A, %B %d, %Y at %H:%M:%S")
        augmented_query = f"{query}\n\nCurrent date, time and timezone: {formatted_date} ({local_timezone})"
        logger.info(f"Sending query to Gemini: {query}")
        print(f"\nCurrent date, time and timezone: {formatted_date} ({local_timezone})")
        
        # Initialize conversation history
        messages = previous_messages.copy() if previous_messages else []
        messages.append({"role": "user", "content": query})
        
        try:
            # Convert MCP tools to Gemini format
            gemini_tools = self._convert_tools_to_gemini_format()
            tools = genai_types.Tool(function_declarations=gemini_tools)
            
            # Prepare chat history
            chat_history = self._prepare_gemini_chat_history(messages)
            
            # Create a chat session with the system prompt
            chat = genai.GenerativeModel(
                model_name=self.model,
                generation_config={"temperature": 0.2},
                tools=[tools],
                system_instruction=SYSTEM_PROMPT
            ).start_chat(history=chat_history)
            
            # Send the augmented query to Gemini
            response = chat.send_message(augmented_query)
            
            # Process the response
            final_text = []
            updated_messages = messages.copy()
            
            final_text, updated_messages = await self._process_gemini_response(
                response, final_text, updated_messages
            )
            
            # Join the final text parts
            response_text = "\n".join(final_text)
            
            return response_text, updated_messages
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"I encountered an error while processing your request: {str(e)}", messages
    
    async def cleanup(self):
        """
        Clean up resources and close connections.
        """
        if hasattr(self, '_session_context') and self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if hasattr(self, '_streams_context') and self._streams_context:
            await self._streams_context.__aexit__(None, None, None)
        
        logger.info("Calendar Assistant resources cleaned up")


async def chat_loop(assistant):
    """
    Run an interactive chat loop with the calendar assistant.
    
    Args:
        assistant: An initialized CalendarAssistant instance
    """
    previous_messages = []
    print("Google Calendar Assistant")
    print("------------------------")
    print("Type your queries about your calendar or 'quit' to exit.")
    print("Type 'refresh' to clear conversation history.")
    
    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() == "quit":
                break
            
            # Check if the user wants to refresh conversation history
            if query.lower() == "refresh":
                previous_messages = []
                print("Conversation history cleared.")
                continue
        
            # Process the query
            response, previous_messages = await assistant.process_query(query, previous_messages)
            print("\nAssistant:", response)
            
        except Exception as e:
            logger.exception("Error in chat loop")
            print("Error:", str(e))


async def main():
    """
    Main function to run the calendar assistant.
    """
    # Get API key and MCP server URL from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    
    # Check if required environment variables are set
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return
    
    if not mcp_server_url:
        print("Error: MCP_SERVER_URL not found in environment variables.")
        return
    
    # Create the calendar assistant
    assistant = CalendarAssistant(mcp_server_url, gemini_api_key)
    
    try:
        # Connect to the MCP server
        connected = await assistant.connect_to_mcp_server()
        if not connected:
            print("Failed to connect to the MCP server. Please check your connection and try again.")
            return
        
        # Start the chat loop
        await chat_loop(assistant)
        
    finally:
        # Clean up resources
        await assistant.cleanup()
        print("\nGoogle Calendar Assistant closed.")


if __name__ == "__main__":
    asyncio.run(main())
