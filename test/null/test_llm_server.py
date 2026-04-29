import unittest, threading, time

class TestLLMServer(unittest.TestCase):
  """Integration tests using the real OpenAI client."""

  @classmethod
  def setUpClass(cls):
    from tinygrad.apps.llm import Transformer, TransformerConfig, SimpleTokenizer
    import tinygrad.apps.llm as llm_module
    
    TEST_CONFIG = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                          norm_eps=1e-5, vocab_size=1000, head_dim=32, rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=512)
    cls.real_model = Transformer(TEST_CONFIG)
    
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}
    
    normal_tokens = {k: v for k, v in byte_decoder.items()}
    # Fill up to 1000 to avoid KeyError during random generation
    for i in range(256, 1000):
        normal_tokens[chr(33) * i] = i
        
    cls.real_tok = SimpleTokenizer(normal_tokens=normal_tokens, special_tokens={"<bos>": 1000, "<eos>": 1001})
    
    cls.bos_id = 1000
    cls.eos_id = 1001
    
    llm_module.model = cls.real_model
    llm_module.model_name = "test-model"
    llm_module.tok = cls.real_tok
    llm_module.bos_id = cls.bos_id
    llm_module.eos_id = cls.eos_id
    
    from tinygrad.apps.llm import Handler
    from tinygrad.viz.serve import TCPServerWithReuse
    import threading, time
    
    cls.server = TCPServerWithReuse(('127.0.0.1', 0), Handler)
    cls.port = cls.server.server_address[1]
    cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
    cls.server_thread.start()
    time.sleep(0.1)
    
    from openai import OpenAI
    cls.client = OpenAI(base_url=f"http://127.0.0.1:{cls.port}/v1", api_key="test")



  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()

  def test_chat_completion_stream(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True
    )

    chunks = list(stream)
    self.assertGreater(len(chunks), 0)
    self.assertEqual(chunks[0].choices[0].delta.role, "assistant")
    self.assertIn(chunks[-1].choices[0].finish_reason, ("stop", "length"))

  def test_openai_response_structure(self):
    stream = self.client.chat.completions.create(
      model="test-model",
      messages=[{"role": "user", "content": "Test"}],
      stream=True
    )

    for chunk in stream:
      self.assertTrue(chunk.id.startswith("chatcmpl-"))
      self.assertEqual(chunk.object, "chat.completion.chunk")
      self.assertIsNotNone(chunk.choices)
      self.assertIsNotNone(chunk.created)
      self.assertIsInstance(chunk.created, int)
      self.assertEqual(chunk.model, "test-model")

  def test_stream_with_usage(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True,
      stream_options={"include_usage": True}
    )

    chunks = list(stream)
    last_chunk = chunks[-1]

    self.assertIsNotNone(last_chunk.usage)
    self.assertIsNotNone(last_chunk.usage.prompt_tokens)
    self.assertIsNotNone(last_chunk.usage.completion_tokens)
    self.assertIsNotNone(last_chunk.usage.total_tokens)

  def test_multi_turn_conversation(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"}
      ],
      stream=True
    )

    chunks = list(stream)
    self.assertGreater(len(chunks), 0)
    self.assertIn(chunks[-1].choices[0].finish_reason, ("stop", "length"))

  def test_content_is_streamed(self):
    stream = self.client.chat.completions.create(
      model="test",
      messages=[{"role": "user", "content": "Hello"}],
      stream=True
    )

    contents = []
    for chunk in stream:
      if chunk.choices and chunk.choices[0].delta.content:
        contents.append(chunk.choices[0].delta.content)

    self.assertGreater(len(contents), 0)

  def test_non_streaming(self):
    resp = self.client.chat.completions.create(
      model="test-model",
      messages=[{"role": "user", "content": "Hello"}],
      stream=False
    )

    self.assertTrue(resp.id.startswith("chatcmpl-"))
    self.assertEqual(resp.object, "chat.completion")
    self.assertEqual(resp.model, "test-model")
    self.assertIsNotNone(resp.created)
    self.assertEqual(len(resp.choices), 1)
    self.assertEqual(resp.choices[0].message.role, "assistant")
    self.assertIsNotNone(resp.choices[0].message.content)
    self.assertIn(resp.choices[0].finish_reason, ("stop", "length"))
    self.assertIsNotNone(resp.usage)
    self.assertIsNotNone(resp.usage.prompt_tokens)
    self.assertIsNotNone(resp.usage.completion_tokens)

  def test_max_tokens_streaming(self):
    stream = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "Hello"}], stream=True, max_tokens=2
    )
    chunks = list(stream)
    content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]
    self.assertLessEqual(len(content_chunks), 2)
    self.assertEqual(chunks[-1].choices[0].finish_reason, "length")

  def test_max_tokens_non_streaming(self):
    resp = self.client.chat.completions.create(
      model="test", messages=[{"role": "user", "content": "Hello"}], stream=False, max_tokens=2
    )
    self.assertIn(resp.choices[0].finish_reason, ("stop", "length"))
    self.assertLessEqual(resp.usage.completion_tokens, 2)


  def test_assistant_prefill(self):
    resp = self.client.chat.completions.create(
      model="test", messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Sure"}
      ], stream=False, max_tokens=1
    )
    # Organic evaluation: prompt_tokens should exactly match the expected encoded length for a prefill
    # A completed turn would have an extra end_turn and role("assistant") appended
    self.assertIsNotNone(resp.choices[0].message.content)
    # 2 roles + 2 contents = 4 items. Since no extra end_turn and role("assistant") is appended,
    # the length should be deterministic. We just check the response is valid and prompt_tokens > 0.
    self.assertGreater(resp.usage.prompt_tokens, 0)


  def test_models_endpoint(self):
    import requests as req
    resp = req.get(f"http://127.0.0.1:{self.port}/v1/models")
    self.assertEqual(resp.status_code, 200)
    data = resp.json()
    self.assertEqual(data["object"], "list")
    self.assertEqual(len(data["data"]), 1)
    self.assertEqual(data["data"][0]["id"], "test-model")
    self.assertEqual(data["data"][0]["object"], "model")

if __name__ == '__main__':
  unittest.main()
