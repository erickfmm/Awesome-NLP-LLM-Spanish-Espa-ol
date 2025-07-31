# Awesome-NLP-LLM-Spanish-Espa-ol
Listado de recursos, datasets, modelos, cursos y otros relacionados al NLP y LLM en Español

Si quieres cursos, libros y tutoriales aprobados (y probados) por mi: [Ruta de Aprendizaje](/ruta-de-aprendizaje.md)

## Otros Awesome similares
* [LLMs In Spanish](https://github.com/drcaiomoreno/LLMsInSpanish)
* [Lacuna Fund](https://lacunafund.org/language-resources/): Recursos varios
* [SomosNLP](https://huggingface.co/somosnlp): Comunidad con hartos Datasets y modelos en Español
* [PrevenIA](https://huggingface.co/PrevenIA): Fundado por el Ministerio de Salud de España, herramientas para prevenir el sui**dio, modelos y datasets

## LLM en español
* [LINCE ZERO](https://huggingface.co/clibrain/lince-zero): Modelo español simple (ZERO), hay una versión FULL que hay que solicitarla
* Projecte AINA (Español y catalán)
  * [Flor 1.3B-Instructed](https://huggingface.co/projecte-aina/FLOR-1.3B-Instructed)
  * [Flor 6.3B](https://huggingface.co/projecte-aina/FLOR-6.3B)
  * [Aguila 7B](https://huggingface.co/projecte-aina/aguila-7b): Basado en Falcon7B
* [Salamandra](https://huggingface.co/collections/BSC-LT/salamandra-66fc171485944df79469043a): por BSC
* [LLM Latino](https://www.latamgpt.org/) (aún es un proyecto en curso, no han lanzado su primer modelo)

### LLM multi idioma (incluye español)
* Meta/Facebook ([Meta Llama](https://www.llama.com/llama-downloads/))
  * [Llama 2](https://huggingface.co/collections/meta-llama/metas-llama2-models-675bfd70e574a62dd0e40541): 7B, 13B, 70B
  * [CodeLlama](https://huggingface.co/collections/meta-llama/code-llama-family-661da32d0a9d678b6f55b933): 7B, 13B, 34B, 70B
  * [Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6): 8B, 70B
  * [LLama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f): 8B, 70B, 405B
  * [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf): 1B, 3B, 11B (con Vision), 90B (con Visión)
  * [Llama 3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct): 70B
* Mistral AI
  * [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
  * [Ministral 8B](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)
  * Mixtral 8x7B
  * [Mistral Large (123B)](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407)
  * [Mistral Nemo (12B)](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
  * [Mixtral 8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)
  * [Magistral-Small](https://huggingface.co/mistralai/Magistral-Small-2507): 24B, con razonamiento
  * [Ministral](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410): Versiones de 3B y 8B, creado para ser ajustado para tareas específicas
* Microsoft
  * [Phi 3.5 mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
  * [Phi 3.5 MoE](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)
  * [Phi 3.5 Vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
  * [Phi 4 mini](https://huggingface.co/microsoft/Phi-4-mini-instruct): 4B aprox (3.8)
  * [Phi 4 Multimodal (Texto, Imagen, Audio)](https://huggingface.co/microsoft/Phi-4-multimodal-instruct): 6B aprox (5.6)
* DeepSeek:
  * [R1](https://huggingface.co/deepseek-ai/DeepSeek-R1): MoE de 687B en total, Entrenado principalmente en Inglés y Chino, pero en sus últimas versiones han mejorado el rendimiendo multi idioma
  * [Distill Qwen 1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B): 1.5B, Entrenado principalmente en Inglés y Chino, generalmente "piensa" en inglés, su rendimiento es bastante bueno en comparación a la cantidad de parámetros
* Qwen3 (de Alibaba):
  * [Qwen 3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f): Serie de modelos que van desde los 0.6B, 1,7B, 4B, 8B, 14B, 32B, y versiones MoE de 30B-A3B (30B en total, 3B activados en cada predicción), 235B-A22B, y versiones actualizadas de los últimos dos modelos MoE
  * [Qwen Coder](https://huggingface.co/collections/Qwen/qwen3-coder-687fc861e53c939e52d52d10): Modelos orientados a la programación, son las versiones MoE finetuneadas
* [Molmo](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)
* [EuroLLM](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct): Por utter project, consorcio de varias universidades de Alemania
* [Apollo](https://huggingface.co/collections/FreedomIntelligence/apollomoe-and-apollo2-670ddebe3bb1ba1aebabbf2c): Profesor de una universidad China
* [Smol LM 3 (HuggingFace)](https://huggingface.co/HuggingFaceTB/SmolLM3-3B): LLM pequeño de Huggingface, bastante bueno.

## Modelos BERT, BART y Transformers en Español
* Departamento de Ciencias de la Computación Universidad de Chile (DCC Uchile)
  * [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased): Modelo BERT Finetuneado de BERT-base (original de Google) con 768 dimensiones, entrenado en MASKED LANGUAGE y finetuneado en varias tareas más
  * [Patana](https://huggingface.co/dccuchile/patana-chilean-spanish-bert): Finetunning de BETO con texto de Chile
  * [Tulio](https://huggingface.co/dccuchile/tulio-chilean-spanish-bert): Finetuning de BETO con texto de Chile y libros en Español.
  * [Versiones livianas](https://github.com/dccuchile/lightweight-spanish-language-models)
* [BETO finetuned on XNLI](https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli)
* [BETO finetuned en summarization](https://huggingface.co/mrm8488/bert2bert_shared-spanish-finetuned-summarization)
* [Bertin Project](https://huggingface.co/bertin-project): Proyecto con datasets para crear un BERT en español. y modelos GPT.
* [Qwen3 Embeddings](https://huggingface.co/collections/Qwen/qwen3-embedding-6841b2055b99c44d9a4c371f): Modelos LLM para generar Embeddings (Similar a NV-EMbed), genera un arreglo de [n_tokens * 1024] donde 1024 es la dimensión de embedding. Por ahora no he visto versiones pooleadas para Sentence Embedding :(
* [Qwen3 Reranker](https://huggingface.co/collections/Qwen/qwen3-reranker-6841b22d0192d7ade9cdefea): Recibe una tarea (un texto indicando el objetivo de la búsqueda semántica), una serie de queries (frases) emparejadas con una serie de documentos (resultados), y les da un score (puntaje).

### Modelos de Embeddings para Sentence Similarity y Semantic Search
* NV-Embed: [v1](https://huggingface.co/nvidia/NV-Embed-v1) y [v2](https://huggingface.co/nvidia/NV-Embed-v2): 7B, Generador de Embeddings basado en un LLM Mistral (multi idioma) multi tarea (prompted). Ventana de 4096 tokens.
* [BETO FINETUNED en Sentence Similarity](https://huggingface.co/hiiamsid/sentence_similarity_spanish_es): 110M, entrenado con dataset STS traducido con DeepL. Ventana de 512 tokens.
* [ModernBert finetuneado](https://huggingface.co/mrm8488/modernbert-embed-base-ft-sts-spanish-matryoshka-768-64): 149M, entrenado con STS traducido al español augmented (dataset privado) Ventana de 8192 tokens.
* [JinaAI](https://huggingface.co/jinaai/jina-embeddings-v3): 572M, Multi idioma, multi tarea query-retrieval. Ventana de 8192 tokens.
* [LaBSE](https://huggingface.co/sentence-transformers/LaBSE): 471M, Multi idioma, entrenado principalmente para tareas de parallel sentences (traducciones). Ventana de 256 tokens.
* [BGE-M3](https://huggingface.co/BAAI/bge-m3): Multidioma. Ventana de 8192 tokens.

## Named Entity Recognition
* [Wikineural](https://huggingface.co/Babelscape/wikineural-multilingual-ner): Multiidioma
* [NER más usado en español, con mejor ACC](https://huggingface.co/MMG/xlm-roberta-large-ner-spanish)
* [Detección de información personal por token](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information): En 4 idiomas

## Traducción
* nllb-200 (Facebook): Versiones [1.3B](https://huggingface.co/facebook/nllb-200-1.3B) y [3.3B](https://huggingface.co/facebook/nllb-200-3.3B) y versiones distilled
* [Seamless (Facebook)](https://huggingface.co/facebook/seamless-m4t-v2-large): Audio a texto, texto a texto
* Helsinki Opus [Inglés a Español](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) y [Español a Inglés](https://huggingface.co/Helsinki-NLP/opus-mt-es-en)

## Speech Recognition
* [Whisper (v3 large turbo)](https://huggingface.co/openai/whisper-large-v3-turbo): Modelo de OpenAI, versiones [small](https://huggingface.co/openai/whisper-small), [tiny](https://huggingface.co/openai/whisper-tiny), también hay versiones [rápidas](https://huggingface.co/Systran/faster-whisper-large-v3)
* [Canary (Nvidia)](https://huggingface.co/collections/nvidia/canary-65c3b83ff19b126a3ca62926): Modelos de Nvidia en 180M y 1B (flash y normal), también [versión basada en Qwen](https://huggingface.co/nvidia/canary-qwen-2.5b)
* [Parakeet (Nvidia y Projecte Aina)](https://huggingface.co/projecte-aina/parakeet-rnnt-1.1b_cv17_es_ep18_1270h): Modelo de Nvidia finetuneado por la gente de Projecte Aina (Cataluña) para español
* [MMS 1B (Facebook)](https://huggingface.co/facebook/mms-1b-all): entrenado en 1162 idiomas
* [Seamless (Facebook)](https://huggingface.co/facebook/seamless-m4t-v2-large): Entrenado en 🎤 101 languages for speech input. 💬 96 Languages for text input/output. 🔊 35 languages for speech output., sirve para:
  * Speech-to-speech translation (S2ST)
  * Speech-to-text translation (S2TT)
  * Text-to-speech translation (T2ST)
  * Text-to-text translation (T2TT)
  * Automatic speech recognition (ASR)
* [Phi4 Multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct): LLM multimodal que entiende y puede transcribir audio
* [Voxtral (Mistral)](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507): LLM Multimodal para audio, entrenado en 8 idiomas, puede entender audio o transcribirlo, según el prompt que le des y la temperatura

## Texto To Speech
* [MMS (Facebook)](https://huggingface.co/facebook/mms-tts-spa): Español
* [Bark (Suno)](https://huggingface.co/suno/bark): Multiidioma
* [Zonoz v0.1-Hybrid](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid): 1.65B, Multiidioma
* [Kokoro 1.0](https://huggingface.co/hexgrad/Kokoro-82M): 82M, modelo pequeñísimo. 8 idiomas y 54 voces, incluye español.
* [Seamless (Facebook)](https://huggingface.co/facebook/seamless-m4t-v2-large): Entrenado en 🎤 101 languages for speech input. 💬 96 Languages for text input/output. 🔊 35 languages for speech output., sirve para:
  * Speech-to-speech translation (S2ST)
  * Speech-to-text translation (S2TT)
  * Text-to-speech translation (T2ST)
  * Text-to-text translation (T2TT)
  * Automatic speech recognition (ASR)

## Librerías para usar NLP en español
* [Spacy](https://spacy.io/models/es): Contiene 3 modelos basados en VectorStores y un modelo basado en Transformers.
  * tok2vec
  * morphologizer
  * parser
  * senter
  * attribute_ruler
  * lemmatizer
  * ner
* [NLTK](https://www.nltk.org/): Principalmente en inglés, también tiene modelos en español
  * Stopwords
  * Stemming
* [Pysentimiento](https://github.com/pysentimiento/pysentimiento)
  * Sentiment Analysis	es, en, it, pt
  * Hate Speech Detection	es, en, it, pt
  * Irony Detection	es, en, it, pt
  * Emotion Analysis	es, en, it, pt
  * NER & POS tagging	es, en
  * Contextualized Hate Speech Detection	es
  * Targeted Sentiment Analysis
* [Gensim](https://radimrehurek.com/gensim/): Para hacer LDA (Análisis de tópicos).
* [pyLDAVis](https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know): Para visualizar los tópicos LDA generados con gensim
* [BERTopic](https://maartengr.github.io/BERTopic/index.html): Para generar tópicos usando BERT
* LLM
  * Ejecutar, cuantizar
    * [transformers (HuggingFace)](https://huggingface.co/docs/transformers/index): Para ejecutar, permite ejecutar cualquier modelo en estructura HF desde HuggingFace (cientos de miles o más), son modelos sin cuantizar en general
    * [Llamacpp](https://github.com/ggerganov/llama.cpp): Permite ejecutar y cuantizar formato GGUF
    * [ONNX](https://github.com/onnx/onnx): Optimizado para on-device mobile, cuantizados o no
    * [Ollama](https://ollama.com/): Genera un servidor web con una api para llamar a diferentes llm, ahora trae un cliente de escritorio también
  * RAG
    * [Llama Index](https://www.llamaindex.ai/)
    * [LangChain](https://www.langchain.com/)
    * [Haystack](https://haystack.deepset.ai/): Soporta agentes también
  * Agentesc
    * [Camel](https://github.com/camel-ai/camel): Los primeros
    * [Haystack](https://haystack.deepset.ai/)
    * [LangGraph](https://www.langchain.com/langgraph): Difícil de usar, fue la pionera
    * [LangFlow](https://github.com/langflow-ai/langflow)
    * [Crew AI](https://www.crewai.com/): Bastante fácil de usar, usa código python o yaml
    * [Crew AI GUI QT (LangGraph)](https://github.com/LangGraph-GUI/CrewAI-GUI-Qt): Interfaz de escritorio para construir agentes de crewai
    * [Autogen (Microsoft)](https://microsoft.github.io/autogen/stable//index.html): Está dentro de las líderes del sector
    * [ADK (Google)](https://google.github.io/adk-docs/): Más reciente, se integra con otros frameworks como langchain y crewai, agnóstico pero optimizado para gemini
    * [SmolAgents (HuggingFace)](https://github.com/huggingface/smolagents): De HuggingFace, agentes y muchas tools, muy fácil de usar y simple pero para prototipos y desarrollo rápido)
    * [Pydantic AI](https://github.com/pydantic/pydantic-ai): Agentes con contratos fuertes, con salidas estructuradas
    * [Eliza](https://github.com/ai16z/eliza)
    * [Spring AI (Alibaba)](https://github.com/alibaba/spring-ai-alibaba): Para Java
  * Memoria
    * [Agno](https://github.com/agno-agi/agno): Incluye agentes, memoria, tool calling, es fácil de usar
    * [Mem0](https://github.com/mem0ai/mem0): Sistema de memoria, de lo mejor y más completo para memoria en llm
  * No-code o Low-code
    * [Langflow](https://www.langflow.org/)
    * [LangGraph studio](https://github.com/langchain-ai/langgraph-studio)
    * [Pyspur](https://github.com/PySpur-Dev/pyspur): Intuitivo con muchas características
 * Finetunning (Full,PEFT,MEFT, etc)
   * [Ludwig](https://ludwig.ai/latest/): Finetunear diferentes tipos de modelos para diferentes tareas, a través de archivos de configuración yaml (low code)
   * [Unsloth](https://unsloth.ai/): Trae optimizaciones de memoria y procesamiento, tiene versión gratis open y versión premium. hay [notebooks de ejemplo](https://github.com/unslothai/notebooks)
   * [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory): Tiene interfaz gráfica y es fácil de usar (incluye optimizaciones de unsloth free y otras)
   * [AutoTrain (HuggingFace)](https://huggingface.co/autotrain): Fácil de usar
   * [Nanotron (HuggingFace)](https://github.com/huggingface/nanotron): Con paralelismo 3D
   * [trl (HuggingFace)](): Para entrenamiento de aprendizaje por refuerzo (reinforcement learning) con distintas técnicas de RLHF como SFT, PPO, DPO
   * [TorchTune (PyTorch)](https://pytorch.org/blog/torchtune-fine-tune-llms/): Más versátil, pero por lo mismo puede ser más difícil
   * [Oumi](https://github.com/oumi-ai/oumi): Para preentrenar o finetunear

### Herramientas de observabilidad
* [Opik](https://github.com/comet-ml/opik): Incluye dashboard
* [Phoenix](https://github.com/Arize-ai/phoenix): Incluye dashboard
* [Openllmetry](https://github.com/traceloop/openllmetry): Emite trazas de llm en formato opentelemetry, lo cual le permite conectarse a casi cualquier herramienta de observabilidad de logs o de monitoreo, como kibanam, grafana, openobserve, entre otros, tiene muchas integraciones de modelos
* [Agentops](https://github.com/AgentOps-AI/agentops): Se integra con frameworks de agentes (CrewAI, CamelAI, Autogen, LangGraph), tiene más de mil evaluaciones, y guarda los flujos y tiene dashboard
* [Langwatch](https://github.com/langwatch/langwatch): Incluye Dashboard, y montón de herramientas como human in the loop, evaluación, datasets, trazas y workflows, etc
* [DeepEval](https://github.com/confident-ai/deepeval): Incluye dashboard, centrado en la evaluación y benchmarks, no tanto en observabilidad y monitoreo, aunque también se puede usar con ese fin
* [Dify](https://github.com/langgenius/dify): Framework todo en uno para fácil deploy
* [Tensorzero](https://github.com/tensorzero/tensorzero): Framework todo en uno para fácil deploy

### Preprocesamiento (de pdf u otro a markdown/json)
* [OlmoOCR](https://github.com/allenai/olmocr): Pasa pdf, png, jpeg a Markdown, basado en un VLM de 7B finetuneado, requiere una GPU con 20G de memoria (al menos)
* [Markitdown (Microsoft)](https://github.com/microsoft/markitdown): Herramienta para pasar de Office, Youtube, EPUB y PDF a Markdown
* [Docling (IBM)](https://github.com/docling-project/docling): Herramienta que se puede usar as-is y usa una heurística para pasar de PDF y otros a markdown/json/html/txt, o usar modelos VLM, desde smoldocling de 360M hasta Granite de 3.3B y 7B, especializados en procesamiento de pdfs
* [Marker](https://github.com/datalab-to/marker): Herramienta bastante potente que usa varios modelos "pequeños" para diferentes tareas, haciéndolo más rápido que otras opciones con VLM y pide menos memoria. transforma office, epub y pdf a markdown, también puede usar un llm para corregir el resultado, con varios proveedores disponibles (gemini, ollama, openai, etc)
* [MinerU](https://github.com/opendatalab/MinerU): Requiere mínimo 16GB recomendado 32GB, cpu o gpu o mps
* [MegaParser](https://github.com/QuivrHQ/MegaParse): Usa api de openai o anthropic, y tesseract para ocr, lo que lo hace liviano de ejecutarse pero necesitas tener la api.

## VectorStores
### Word Vectors
* [Word Vector 3B Words](https://www.kaggle.com/datasets/julianusugaortiz/spanish-3b-words-word2vec-embeddings)
* [Word vector 1B words](https://www.kaggle.com/datasets/rtatman/pretrained-word-vectors-for-spanish)
* [DCC Uchile](https://github.com/dccuchile/spanish-word-embeddings)

### Sentence Vectors
* [Spanish Sentence Embeddings](https://github.com/BotCenter/spanish-sent2vec): Calculadas usando el dataset SUC y el programa [sent2vec](https://github.com/epfml/sent2vec)

## Datasets
* [Coleccion de datasets en español por metatext](https://metatext.io/datasets-list/spanish-language)
* [Bertin Project](https://huggingface.co/bertin-project): Proyecto para crear GPT y Bert en español. Tienen datasets de tipo Alpaca y similar (pares pregunta - respuesta para crear modelos Instruct)
* [Red Pajama](https://huggingface.co/datasets/latam-gpt/red_pajama_es_hq): Dataset ENORME en español de texto no estructurado para entrenar un LLM, con un score de 2 a 5 según su calidad académica


### No etiquetado
* DCC Uchile
  * [Spanish Unanotated Corpora (SUC)](https://github.com/josecannete/spanish-corpora): Usado para entrenar BETO
  * [Spanish Books](https://huggingface.co/datasets/jorgeortizfuentes/spanish_books)
  * [Chilean Spanish Corpus](https://huggingface.co/datasets/jorgeortizfuentes/chilean-spanish-corpus)
  * [Universal Spanish Chilean Corpus](https://huggingface.co/datasets/jorgeortizfuentes/universal_spanish_chilean_corpus): La suma de los dos anteriores (aparentemente)
* [Wikihow en español](https://huggingface.co/datasets/daqc/wikihow-spanish)
* [Wikipedia as TXT](https://huggingface.co/datasets/daqc/wikipedia-txt-spanish)
* [Wikipedia oficial](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.es)
* [Wiktionary](https://huggingface.co/datasets/carloscapote/es.wiktionary.org): Diccionario de la Wikipedia en español
* [9322 letras de rap en español](https://www.kaggle.com/datasets/smunoz3801/9325-letras-de-rap-en-espaol/data)
* [Poemas en español](https://huggingface.co/datasets/andreamorgar/spanish_poetry)
* [Wikibooks (Multi idioma)](https://www.kaggle.com/datasets/dhruvildave/wikibooks-dataset)
* [15k Noticias en español](https://huggingface.co/datasets/BrauuHdzM/noticias-en-espanol)
* [TEDx En español](https://huggingface.co/datasets/ittailup/tedx_spanish)
* Obtenidos mediante Scraping a datos públicos MINEDUC (Ministerio de Educación de Chile)
  * [Proyectos Educativos Institucionales (PEI)](https://www.kaggle.com/datasets/erickfmm/education-pei-pdf): En pdf
  * [Reglamentos de Convivencia](https://www.kaggle.com/datasets/erickfmm/education-reglamento-convivencia-pdf): En pdf
  * [Reglamentos de Evaluación](https://www.kaggle.com/datasets/erickfmm/education-reglamentos-de-evaluacin-pdf): En pdf

### Etiquetado
* [Datasets QA en HF](https://huggingface.co/datasets?task_categories=task_categories:question-answering&language=language:es&sort=trending)
* [HEAR - Hispanic Emotional Accompaniment Responses](https://huggingface.co/datasets/BrunoGR/HEAR-Hispanic_Emotional_Accompaniment_Responses?row=2): Se usaron datos sintéticos para balancear dataset. par pregunta-respuesta para acompañamiento emocional
* [Punta Cana reviews](https://huggingface.co/datasets/beltrewilton/punta-cana-spanish-reviews): útil para sistemas recomendadores u otras tareas
* [Reviews de hoteles de Andalucia](https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced)
* [Reviews de comida peruana](https://www.kaggle.com/datasets/lazaro97/peruvian-food-reviews)
* [Reseñas IMDB TRADUCIDAS al español](https://www.kaggle.com/datasets/luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish)
* [XNLI (Inferencia dada una premisa y hipótesis), Multiidioma](https://huggingface.co/datasets/facebook/xnli)
* [HateSpeech (Multi idioma)](https://www.kaggle.com/datasets/wajidhassanmoosa/multilingual-hatespeech-dataset)
* [STS (Sentence Similarity) Traducido con DeepL a varios idiomas](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt)
* [DWUG ES: Diachronic Word Usage Graphs for Spanish](https://zenodo.org/records/6433667#.YmGU7i8lP0o)
* [Diagnósticos médicos en español, y si son dentales o no](https://huggingface.co/datasets/fvillena/spanish_diagnostics)

### Mapudungun
* [HuggingFace](https://huggingface.co/datasets?language=language:arn&sort=trending)
* [Mapudungun corpus cleaned](https://github.com/mingjund/mapudungun-corpus)
* [Corpus of Historical Mapudungun (CHM)](https://benmolineaux.github.io/)
