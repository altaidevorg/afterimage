Providers
=========

.. automodule:: afterimage.providers
   :members:
   :undoc-members:
   :show-inheritance:

Document Providers
------------------

.. automodule:: afterimage.providers.document_providers
   :members:
   :undoc-members:
   :show-inheritance:

LLM Providers
-------------

.. automodule:: afterimage.providers.llm_providers
   :members:
   :undoc-members:
   :show-inheritance:

Embedding providers
-------------------

Async text embeddings (OpenAI-compatible APIs, Gemini, and local SentenceTransformer
via a process pool). Public types are re-exported on the ``afterimage`` package.

.. autoclass:: afterimage.EmbeddingProvider
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: afterimage.OpenAIEmbeddingProvider
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: afterimage.GeminiEmbeddingProvider
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: afterimage.ProcessEmbeddingProvider
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: afterimage.EmbeddingProviderFactory
   :members:
   :undoc-members:
   :show-inheritance:
