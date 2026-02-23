import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch
from llama_index.core.llms import ChatMessage, MessageRole
from droidrun.agent.utils.custom_llm import SimpleCustomLLM

class TestSimpleCustomLLM(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.api_url = "http://localhost:8080/predict"
        self.llm = SimpleCustomLLM(
            api_url=self.api_url,
            prompt_key="input_text",
            response_key="output"
        )

    @patch("requests.post")
    def test_complete_sync(self, mock_post):
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "Test response sync"}
        mock_post.return_value = mock_response

        response = self.llm.complete("Hello")
        
        self.assertEqual(response.text, "Test response sync")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"], {"input_text": "Hello"})

    @patch("requests.post")
    async def test_acomplete_async(self, mock_post):
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "Test response async"}
        mock_post.return_value = mock_response

        response = await self.llm.acomplete("Hello Async")
        
        self.assertEqual(response.text, "Test response async")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"], {"input_text": "Hello Async"})

    @patch("requests.post")
    async def test_achat_flattening(self, mock_post):
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "I am looking at the screen."}
        mock_post.return_value = mock_response

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are an assistant."),
            ChatMessage(role=MessageRole.USER, content="What do you see?"),
        ]
        
        response = await self.llm.achat(messages)
        
        self.assertEqual(response.message.content, "I am looking at the screen.")
        
        # Verify prompt flattening
        args, kwargs = mock_post.call_args
        prompt = kwargs["json"]["input_text"]
        self.assertIn("SYSTEM: You are an assistant.", prompt)
        self.assertIn("USER: What do you see?", prompt)
        self.assertTrue(prompt.strip().endswith("ASSISTANT:"))

    @patch("requests.post")
    async def test_astream_chat_simulation(self, mock_post):
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output": "Streamed response"}
        mock_post.return_value = mock_response

        messages = [ChatMessage(role=MessageRole.USER, content="Stream me")]
        
        chunks = []
        async for chunk in self.llm.astream_chat(messages):
            chunks.append(chunk)
            
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].message.content, "Streamed response")
        self.assertEqual(chunks[0].delta, "Streamed response")

if __name__ == "__main__":
    unittest.main()
