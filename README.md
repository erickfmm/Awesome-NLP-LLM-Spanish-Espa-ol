# Awesome NLP & LLM en Español
Listado de recursos, datasets, modelos, cursos y otros relacionados al NLP y LLM en Español

Si quieres cursos, libros y tutoriales aprobados (y probados) por mi: [Ruta de Aprendizaje](/ruta-de-aprendizaje.md)

---

## 📋 Índice

- [🌟 Otros Awesome Similares](#-otros-awesome-similares)
- [🤖 LLM en Español](#-llm-en-español)
  - [🌍 LLM Multi Idioma (incluye español)](#-llm-multi-idioma-incluye-español)
- [🔤 Modelos BERT, BART y Transformers en Español](#-modelos-bert-bart-y-transformers-en-español)
  - [📊 Modelos de Embeddings para Sentence Similarity y Semantic Search](#-modelos-de-embeddings-para-sentence-similarity-y-semantic-search)
- [🏷️ Named Entity Recognition](#️-named-entity-recognition)
- [🌐 Traducción](#-traducción)
- [🎤 Reconocimiento de Voz (ASR)](#-reconocimiento-de-voz-asr)
- [🔊 Texto to Speech (TTS)](#-texto-to-speech-tts)
- [🛠️ Librerías para Usar NLP en Español](#️-librerías-para-usar-nlp-en-español)
  - [🤝 Agentes](#-agentes)
  - [🗄️ Bases de Datos de Embeddings y Visualizadores](#️-bases-de-datos-de-embeddings-y-visualizadores)
  - [🔭 Herramientas de Observabilidad](#-herramientas-de-observabilidad)
  - [📄 Preprocesamiento (de PDF u otro a Markdown/JSON)](#-preprocesamiento-de-pdf-u-otro-a-markdownjson)
- [📦 VectorStores Precomputados](#-vectorstores-precomputados)
  - [Word Vectors](#word-vectors)
  - [Sentence Vectors](#sentence-vectors)
- [📏 Evaluación y Métricas](#-evaluación-y-métricas)
  - [Benchmarks](#benchmarks)
- [📚 Datasets](#-datasets)
  - [No Etiquetado](#no-etiquetado)
  - [Etiquetado](#etiquetado)
  - [Mapudungun](#mapudungun)
- [🗂️ Mis Datasets (erickfmm)](#️-mis-datasets-erickfmm)

---

## 🌟 Otros Awesome Similares
* [LLMs In Spanish](https://github.com/drcaiomoreno/LLMsInSpanish)
* [Lacuna Fund](https://lacunafund.org/language-resources/): Recursos varios
* [SomosNLP](https://huggingface.co/somosnlp): Comunidad con hartos Datasets y modelos en Español
* [PrevenIA](https://huggingface.co/PrevenIA): Fundado por el Ministerio de Salud de España, herramientas para prevenir el sui**dio, modelos y datasets
* [Corpus: Evaluation datasets for ES & LATAM (SomosNLP)](https://huggingface.co/collections/somosnlp/corpus-evaluation-datasets-for-es-and-latam): Batería de datasets para evaluar LLMs en español y variantes (oro para benchmarks)
* [Corpus: Instructions in Spanish and related languages (SomosNLP)](https://huggingface.co/collections/somosnlp/corpus-instructions-in-spanish-and-related-languages): Colección de datasets de instrucciones/SFT para afinar modelos en español
* [Instruction-Tuned Models ES (SomosNLP)](https://huggingface.co/collections/somosnlp/instruction-tuned-models-es): Curación de modelos instruct en español y lenguas cercanas (ahorra horas de búsqueda)

## 🤖 LLM en Español
* [LINCE ZERO](https://huggingface.co/clibrain/lince-zero): Modelo español simple (ZERO), hay una versión FULL que hay que solicitarla
* Projecte AINA (Español y catalán)
  * [Flor 1.3B-Instructed](https://huggingface.co/projecte-aina/FLOR-1.3B-Instructed)
  * [Flor 6.3B](https://huggingface.co/projecte-aina/FLOR-6.3B)
  * [Aguila 7B](https://huggingface.co/projecte-aina/aguila-7b): Basado en Falcon7B
* [Salamandra](https://huggingface.co/collections/BSC-LT/salamandra-66fc171485944df79469043a): Familia de modelos por BSC-LT, muy buenos para español y otras lenguas europeas
  * [salamandra-7b-instruct](https://huggingface.co/BSC-LT/salamandra-7b-instruct): Buen equilibrio calidad/costo, ideal para chat y tareas generales en ES
  * [salamandra-2b-instruct](https://huggingface.co/BSC-LT/salamandra-2b-instruct): Versión ligera para correr y prototipar con menos recursos
  * [Salamandra-VL-7B-2512](https://huggingface.co/BSC-LT/Salamandra-VL-7B-2512): Versión multimodal (visión+texto) potente, útil para investigación y demos (ojo con la licencia)
* [IberianLLM](https://huggingface.co/IberianLLM): Familia de modelos open source para las lenguas ibéricas (español, portugués, catalán, euskera, gallego), desarrollados por el BSC
  * [Iberian-7B-instruct-v1](https://huggingface.co/IberianLLM/Iberian-7B-instruct-v1): 7B, orientado a chat, traducción y tareas generales en lenguas ibéricas
* [ALIA-40B](https://huggingface.co/BSC-LT/ALIA-40b): 40B, primer modelo público y abierto de España, entrenado desde cero en 35 idiomas europeos (con español y lenguas cooficiales) y código, licencia Apache 2.0
* [LLM Latino / LatamGPT](https://www.latamgpt.org/) - [Modelo en HuggingFace](https://huggingface.co/latam-gpt): Primera iniciativa latinoamericana de LLM open source, liderada por el CENIA de Chile. Primer modelo: Llama 3.1 finetuneado con datos culturales de +15 países de América Latina
  * [Llama-3.1-70B-LatamGPT-SFT-1.0](https://huggingface.co/latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0): 70B, finetuneado con datos latinoamericanos, multiidioma (español, portugués e inglés), enfocado en los matices culturales de la región

### 🌍 LLM Multi Idioma (incluye español)
* Meta/Facebook ([Meta Llama](https://www.llama.com/llama-downloads/))
  * [Llama 2](https://huggingface.co/collections/meta-llama/metas-llama2-models-675bfd70e574a62dd0e40541): 7B, 13B, 70B
  * [CodeLlama](https://huggingface.co/collections/meta-llama/code-llama-family-661da32d0a9d678b6f55b933): 7B, 13B, 34B, 70B
  * [Llama 3](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6): 8B, 70B
  * [LLama 3.1](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f): 8B, 70B, 405B
  * [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf): 1B, 3B, 11B (con Vision), 90B (con Visión)
  * [Llama 3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct): 70B
  * [Llama 4](https://huggingface.co/collections/meta-llama/llama-4-67f2b4c4b6a06be3b3e8e51c): MoE multimodal (texto + imagen), entrenado en 200 idiomas (12 con fine-tuning especializado)
    * [Llama 4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct): MoE 109B total / 17B activados (16 expertos), multimodal, contexto de 10M tokens, ideal por su ventana larguísima
    * [Llama 4 Maverick](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct): MoE 400B total / 17B activados (128 expertos), multimodal, contexto de 1M tokens, muy buen rendimiento multiidioma
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
  * [Phi 4](https://huggingface.co/microsoft/phi-4): 14B, orientado a razonamiento, matemáticas y código, fuerte en STEM, licencia MIT
  * [Phi 4 Reasoning](https://huggingface.co/microsoft/Phi-4-reasoning): 14B, versión finetuneada en tareas de razonamiento profundo, ventana de 32K tokens
* Google:
  * [Gemma 3](https://huggingface.co/collections/google/gemma-3-release): Serie multimodal (imagen+texto) con soporte para 140+ idiomas incluyendo español. Tamaños: 1B (texto), 4B, 12B y 27B (multimodal), ventana de 128K tokens
    * [Gemma 3 27B Instruct](https://huggingface.co/google/gemma-3-27b-it): Versión más grande, excelente rendimiento multiidioma y multimodal
    * [Gemma 3 12B Instruct](https://huggingface.co/google/gemma-3-12b-it): Buen equilibrio calidad/costo
    * [Gemma 3 4B Instruct](https://huggingface.co/google/gemma-3-4b-it): Opción ligera con multimodalidad
    * [Gemma 3 1B Instruct](https://huggingface.co/google/gemma-3-1b-it): Ultra ligero, solo texto y solo inglés
  * [Gemma 4](https://huggingface.co/collections/google/gemma-4): Serie multimodal con soporte para 140+ idiomas incluyendo español. Modelos densos y MoE, soportan texto, imagen y audio (modelos pequeños). Licencia Apache 2.0
    * [Gemma 4 E2B IT](https://huggingface.co/google/gemma-4-e2b-it): ~2.3B activados (5.1B con embeddings), multimodal (texto, imagen, audio), ventana 128K tokens. Orientado a dispositivos móviles y edge
    * [Gemma 4 E4B IT](https://huggingface.co/google/gemma-4-e4b-it): ~4.5B activados (8B con embeddings), multimodal (texto, imagen, audio), ventana 128K tokens. Buen equilibrio para laptops
    * [Gemma 4 26B A4B IT](https://huggingface.co/google/gemma-4-26b-a4b-it): MoE, 26B total / 4B activados, multimodal (texto, imagen), ventana 256K tokens. Excelente calidad/costo en GPU consumer
    * [Gemma 4 31B IT](https://huggingface.co/google/gemma-4-31b-it): 31B denso, multimodal (texto, imagen), ventana 256K tokens. Mayor calidad de la familia
* DeepSeek:
  * [V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash): MoE de 284B total / 13B activados, contexto de 1M tokens, licencia MIT
  * [V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro): MoE de 1.6T total / 49B activados, contexto de 1M tokens, licencia MIT
  * [V3](https://huggingface.co/deepseek-ai/DeepSeek-V3): MoE de 671B, código abierto, muy alto rendimiento en razonamiento y código, fuerte en multiidioma
  * [R1](https://huggingface.co/deepseek-ai/DeepSeek-R1): MoE de 687B en total, entrenado principalmente en inglés y chino, pero en sus últimas versiones han mejorado el rendimiento multi idioma
  * [Distill Qwen 1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B): 1.5B, entrenado principalmente en inglés y chino, generalmente "piensa" en inglés, su rendimiento es bastante bueno en comparación a la cantidad de parámetros
* Qwen3 (de Alibaba):
  * [Qwen 3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f): Serie de modelos que van desde los 0.6B, 1,7B, 4B, 8B, 14B, 32B, y versiones MoE de 30B-A3B (30B en total, 3B activados en cada predicción), 235B-A22B, y versiones actualizadas de los últimos dos modelos MoE
  * [Qwen Coder](https://huggingface.co/collections/Qwen/qwen3-coder-687fc861e53c939e52d52d10): Modelos orientados a la programación, son las versiones MoE finetuneadas
  * [Qwen 3.6](https://huggingface.co/collections/Qwen/qwen36): Serie de modelos de nueva generación enfocada en razonamiento y código, ventana de 262K tokens (extensible a 1M)
    * [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B): 27B denso (BF16), supera modelos MoE mucho más grandes en benchmarks de código
    * [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): MoE de 35B total / 3B activados (BF16), muy eficiente en inferencia
* Kimi (Moonshot AI)
  * [Kimi K2 Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking): MoE de 1T total / 32B activados, multiidioma, muy capaz a niveles del estado del arte, orientado a tareas agénticas y razonamiento extendido
  * [Colección Kimi K2](https://huggingface.co/collections/moonshotai/kimi-k2): Incluye versiones Base e Instruct, con capacidades multimodales en modelos K2.5/K2.6
* GLM (Zhipu AI / Tsinghua)
  * [Colección GLM-4](https://huggingface.co/collections/THUDM/glm-4-series): Serie de modelos multiidioma (chino y otros), open source bajo licencia MIT
  * [GLM-4-32B](https://huggingface.co/THUDM/GLM-4-32B-0414): 32B, muy capaz en razonamiento, código y chat, compite con modelos mucho más grandes
  * [GLM-4-9B-Chat](https://huggingface.co/THUDM/glm-4-9b-chat): 9B, buena opción liviana con ventana de 128K tokens
  * [GLM-Z1-32B](https://huggingface.co/THUDM/GLM-Z1-32B-0414): 32B, versión especializada en razonamiento profundo y matemáticas
* NVIDIA
  * [Colección Nemotron](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3): Familia de modelos abiertos de NVIDIA, MoE con enfoque en agentes multi-paso y workflows empresariales
  * [Nemotron-Nano (~30B MoE)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8): MoE de 30B total / 3B activados, rápido, bueno para sub-agentes y despliegue en edge, contexto de 1M tokens
  * [Nemotron-Super (~120B MoE)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8): MoE de 120B total / 12B activados, pensado para workflows multi-agente empresariales
* [Molmo](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)
* [EuroLLM](https://huggingface.co/collections/utter-project/eurollm): Consorcio de universidades alemanas, modelos enfocados en idiomas europeos incluyendo español
* [Apollo](https://huggingface.co/collections/FreedomIntelligence/apollomoe-and-apollo2-670ddebe3bb1ba1aebabbf2c): Profesor de una universidad China
* [Smol LM 3 (HuggingFace)](https://huggingface.co/HuggingFaceTB/SmolLM3-3B): LLM pequeño de Huggingface, bastante bueno.

## 🔤 Modelos BERT, BART y Transformers en Español
* Departamento de Ciencias de la Computación Universidad de Chile (DCC Uchile)
  * [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased): Modelo BERT finetuneado de BERT-base (original de Google) con 768 dimensiones, entrenado en MASKED LANGUAGE y finetuneado en varias tareas más
  * [Patana](https://huggingface.co/dccuchile/patana-chilean-spanish-bert): Finetuning de BETO con texto de Chile
  * [Tulio](https://huggingface.co/dccuchile/tulio-chilean-spanish-bert): Finetuning de BETO con texto de Chile y libros en Español.
  * [Versiones livianas](https://github.com/dccuchile/lightweight-spanish-language-models)
* [BETO finetuned on XNLI](https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli)
* [BETO finetuned en summarization](https://huggingface.co/mrm8488/bert2bert_shared-spanish-finetuned-summarization)
* [RigoBERTa 2.0](https://huggingface.co/IIC/RigoBERTa-2.0): Modelo BERT robusto entrenado en español, excelente para tareas de NLU
* [mmBERT](https://huggingface.co/jhu-clsp): Encoder masivamente multilingüe basado en ModernBERT, entrenado en 1800+ idiomas (incluye español), por JHU-CLSP. Ventana de 8192 tokens. Versiones [small (140M)](https://huggingface.co/jhu-clsp/mmBERT-small) y [base (307M)](https://huggingface.co/jhu-clsp/mmBERT-base). Supera XLM-R en la mayoría de benchmarks multilingüe
* [MrBERT](https://huggingface.co/BSC-LT/MrBERT): Encoder moderno multilingüe de BSC-LT basado en ModernBERT, con versiones especializadas en español/catalán. Ventana de 8192 tokens, usa RoPE y GeGLU. Versiones: [MrBERT-es (español)](https://huggingface.co/BSC-LT/MrBERT-es), [MrBERT multilingüe (35 idiomas)](https://huggingface.co/BSC-LT/MrBERT)
* [EuroBERT](https://huggingface.co/collections/EuroBERT/eurobert): Colección de modelos BERT multilingües europeos con buen soporte para español
* [Bertin Project](https://huggingface.co/bertin-project): Proyecto con datasets para crear un BERT en español. y modelos GPT.
* [Qwen3 Embeddings](https://huggingface.co/collections/Qwen/qwen3-embedding-6841b2055b99c44d9a4c371f): Modelos LLM para generar Embeddings (Similar a NV-EMbed), genera un arreglo de [n_tokens * 1024] donde 1024 es la dimensión de embedding. Por ahora no he visto versiones pooleadas para Sentence Embedding :(
* [Qwen3 Reranker](https://huggingface.co/collections/Qwen/qwen3-reranker-6841b22d0192d7ade9cdefea): Recibe una tarea (un texto indicando el objetivo de la búsqueda semántica), una serie de queries (frases) emparejadas con una serie de documentos (resultados), y les da un score (puntaje).

### 📊 Modelos de Embeddings para Sentence Similarity y Semantic Search
* NV-Embed: [v1](https://huggingface.co/nvidia/NV-Embed-v1) y [v2](https://huggingface.co/nvidia/NV-Embed-v2): 7B, Generador de Embeddings basado en un LLM Mistral (multi idioma) multi tarea (prompted). Ventana de 4096 tokens.
* [BETO finetuned en Sentence Similarity](https://huggingface.co/hiiamsid/sentence_similarity_spanish_es): 110M, entrenado con dataset STS traducido con DeepL. Ventana de 512 tokens.
* [ModernBert finetuneado](https://huggingface.co/mrm8488/modernbert-embed-base-ft-sts-spanish-matryoshka-768-64): 149M, entrenado con STS traducido al español augmented (dataset privado) Ventana de 8192 tokens.
* [JinaAI](https://huggingface.co/jinaai/jina-embeddings-v3): 572M, Multi idioma, multi tarea query-retrieval. Ventana de 8192 tokens.
* [LaBSE](https://huggingface.co/sentence-transformers/LaBSE): 471M, Multi idioma, entrenado principalmente para tareas de parallel sentences (traducciones). Ventana de 256 tokens.
* [BGE-M3](https://huggingface.co/BAAI/bge-m3): Multidioma. Ventana de 8192 tokens.
* [Embedding Gemma 300](https://huggingface.co/google/embeddinggemma-300m): Modelo más avanzado de Google, multiidioma
* [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B): Hay versiones de 4B y más, pero este es más cercano al tamaño estándar, es un LLM que puedes usar como SentenceTransformer, lo que hace internamente es generar el texto dado el prompt que le das (o documento con prompt por defecto) y te genera una respuesta internamente, y te retorna el embedding correspondiente al último token generado.
* [Paraphrase Multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2): 118M, modelo compacto y rápido para similitud semántica multilingüe, buen balance rendimiento-velocidad
* [Paraphrase Multilingual MPNet](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2): Modelo multilingüe muy fuerte para embeddings (muy buen "default" si priorizas calidad)
* [DistilUSE Multilingual v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2): Embeddings multilingües más livianos (512 dims), buena opción si priorizas velocidad/costo

## 🏷️ Named Entity Recognition
* [Wikineural](https://huggingface.co/Babelscape/wikineural-multilingual-ner): Multiidioma
* [NER más usado en español, con mejor ACC](https://huggingface.co/MMG/xlm-roberta-large-ner-spanish)
* [Detección de información personal por token](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information): En 4 idiomas

## 🌐 Traducción
* nllb-200 (Facebook): Versiones [1.3B](https://huggingface.co/facebook/nllb-200-1.3B) y [3.3B](https://huggingface.co/facebook/nllb-200-3.3B) y versiones distilled
* [Seamless (Facebook)](https://huggingface.co/facebook/seamless-m4t-v2-large): Audio a texto, texto a texto
* Helsinki Opus [Inglés a Español](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) y [Español a Inglés](https://huggingface.co/Helsinki-NLP/opus-mt-es-en)
* Tencent Hunyuan HY-MT: Familia de modelos de traducción multilingüe de Tencent, soporta 33 idiomas incluyendo español, con seguimiento de instrucciones de estilo, contexto y terminología
  * [HY-MT1.5 (1.8B)](https://huggingface.co/tencent/HY-MT1.5-1.8B): 1.8B, modelo pequeño que supera APIs comerciales en evaluaciones generales
  * [HY-MT2 (colección)](https://huggingface.co/collections/tencent/hy-mt2): Versión mejorada con mayor calidad y capacidades de instrucción; disponible cuantizado para dispositivos de bajo recurso

## 🎤 Reconocimiento de Voz (ASR)
* [Whisper (v3 large turbo)](https://huggingface.co/openai/whisper-large-v3-turbo): Modelo de OpenAI, versiones [small](https://huggingface.co/openai/whisper-small), [tiny](https://huggingface.co/openai/whisper-tiny), también hay versiones [rápidas](https://huggingface.co/Systran/faster-whisper-large-v3)
* [Canary (Nvidia)](https://huggingface.co/collections/nvidia/canary-65c3b83ff19b126a3ca62926): Modelos de Nvidia en 180M y 1B (flash y normal), también [versión basada en Qwen](https://huggingface.co/nvidia/canary-qwen-2.5b)
* Parakeet (NVIDIA)
  * [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): ASR multilingüe (incluye español), rápido y con timestamps; muy buena opción para transcripción a escala
  * [Colección Parakeet](https://huggingface.co/collections/nvidia/parakeet): Variantes CTC/RNNT/TDT para distintos trade-offs de latencia/precisión
* [Parakeet finetuneado para español (Projecte Aina)](https://huggingface.co/projecte-aina/parakeet-rnnt-1.1b_cv17_es_ep18_1270h): Modelo de Nvidia finetuneado por la gente de Projecte Aina (Cataluña) para español
* [MMS 1B (Facebook)](https://huggingface.co/facebook/mms-1b-all): entrenado en 1162 idiomas
* [Seamless (Facebook)](https://huggingface.co/facebook/seamless-m4t-v2-large): Entrenado en 🎤 101 languages for speech input. 💬 96 Languages for text input/output. 🔊 35 languages for speech output., sirve para:
  * Speech-to-speech translation (S2ST)
  * Speech-to-text translation (S2TT)
  * Text-to-speech translation (T2ST)
  * Text-to-text translation (T2TT)
  * Automatic speech recognition (ASR)
* [Phi4 Multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct): LLM multimodal que entiende y puede transcribir audio
* [Voxtral (Mistral)](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507): LLM Multimodal para audio, entrenado en 8 idiomas, puede entender audio o transcribirlo, según el prompt que le des y la temperatura
* [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602): 4B, versión en tiempo real optimizada para transcripción rápida y eficiente
* [VibeVoice ASR](https://huggingface.co/microsoft/VibeVoice-ASR): Modelo de Microsoft especializado en reconocimiento de voz con baja latencia
* [Auden ASR (AudenAI)](https://huggingface.co/AudenAI): Toolbox open source de audio enfocado en ASR reproducible, modelos para chino, inglés y multilingüe con arquitectura RNN-T/Zipformer; incluye modelos de transcripción, traducción y alineación
  * [auden-asr-zh-en](https://huggingface.co/AudenAI/auden-asr-zh-en): ~170M, ASR bilingüe chino-inglés, entrenado en 245K horas de audio
  * [auden-tta-m10](https://huggingface.co/AudenAI/auden-tta-m10): Modelo multilingüe con capacidades de transcripción, traducción y alineación

## 🔊 Texto to Speech (TTS)
* [MMS (Facebook)](https://huggingface.co/facebook/mms-tts-spa): Español
* [Bark (Suno)](https://huggingface.co/suno/bark): Multiidioma
* [Zonoz v0.1-Hybrid](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid): 1.65B, Multiidioma
* [Kokoro 1.0](https://huggingface.co/hexgrad/Kokoro-82M): 82M, modelo pequeñísimo. 8 idiomas y 54 voces, incluye español.
* [Qwen3 TTS CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice): 1.7B, soporta clonación de voz personalizada, multiidioma con excelente calidad
* [Fun CosyVoice3](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512): 0.5B, modelo compacto con síntesis natural multilingüe
* [Chatterbox (ResembleAI)](https://huggingface.co/ResembleAI/chatterbox): Modelo conversacional de streaming TTS con baja latencia, ideal para aplicaciones en tiempo real
* [VibeVoice Realtime 0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B): 0.5B, optimizado para síntesis en tiempo real con latencia ultra baja
* [VibeVoice 1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B): 1.5B, versión más robusta con mayor calidad de voz, multiidioma
* [Fish S1 Mini](https://huggingface.co/fishaudio/s1-mini): Modelo ligero y eficiente para síntesis de voz con buena calidad en múltiples idiomas
* [Seamless (Facebook)](https://huggingface.co/facebook/seamless-m4t-v2-large): Entrenado en 🎤 101 languages for speech input. 💬 96 Languages for text input/output. 🔊 35 languages for speech output., sirve para:
  * Speech-to-speech translation (S2ST)
  * Speech-to-text translation (S2TT)
  * Text-to-speech translation (T2ST)
  * Text-to-text translation (T2TT)
  * Automatic speech recognition (ASR)

## 🛠️ Librerías para Usar NLP en Español
* [Data dreamer](https://datadreamer.dev/docs/latest/pages/get_started/quick_tour/dataset_augmentation.html) para hacer data augmentation y generar datos sintéticos
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
* Tópicos:
  * [Gensim](https://radimrehurek.com/gensim/): Para hacer LDA (Análisis de tópicos).
  * [pyLDAVis](https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know): Para visualizar los tópicos LDA generados con gensim
  * [BERTopic](https://maartengr.github.io/BERTopic/index.html): Para generar tópicos usando BERT, soporta también contextualizado y diversos métodos, muy completo
  * [Contextualizado](https://github.com/MilaNLProc/contextualized-topic-models)
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
  * Finetuning (Full, PEFT, MEFT, etc.)
    * [Ludwig](https://ludwig.ai/latest/): Finetunear diferentes tipos de modelos para diferentes tareas, a través de archivos de configuración yaml (low code)
    * [Unsloth](https://unsloth.ai/): Trae optimizaciones de memoria y procesamiento, tiene versión gratis open y versión premium. hay [notebooks de ejemplo](https://github.com/unslothai/notebooks)
    * [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory): Tiene interfaz gráfica y es fácil de usar (incluye optimizaciones de unsloth free y otras)
    * [AutoTrain (HuggingFace)](https://huggingface.co/autotrain): Fácil de usar
    * [Nanotron (HuggingFace)](https://github.com/huggingface/nanotron): Con paralelismo 3D
    * [trl (HuggingFace)](https://github.com/huggingface/trl): Para entrenamiento de aprendizaje por refuerzo (reinforcement learning) con distintas técnicas de RLHF como SFT, PPO, DPO
    * [TorchTune (PyTorch)](https://pytorch.org/blog/torchtune-fine-tune-llms/): Más versátil, pero por lo mismo puede ser más difícil
    * [Oumi](https://github.com/oumi-ai/oumi): Para preentrenar o finetunear

### 🤝 Agentes

#### Con código (Python / programático)
* [Camel](https://github.com/camel-ai/camel): Los primeros en investigar agentes multi-agente
* [Haystack](https://haystack.deepset.ai/): También soporta pipelines de RAG
* [LangGraph](https://www.langchain.com/langgraph): Grafos de flujo para agentes, fue pionera aunque puede ser compleja
* [Crew AI](https://www.crewai.com/): Bastante fácil de usar, usa código python o yaml
* [Autogen (Microsoft)](https://microsoft.github.io/autogen/stable//index.html): Está dentro de las líderes del sector
* [ADK (Google)](https://google.github.io/adk-docs/): Más reciente, se integra con otros frameworks como langchain y crewai, agnóstico pero optimizado para gemini
* [SmolAgents (HuggingFace)](https://github.com/huggingface/smolagents): De HuggingFace, agentes y muchas tools, muy fácil de usar y simple, ideal para prototipos y desarrollo rápido
* [Pydantic AI](https://github.com/pydantic/pydantic-ai): Agentes con contratos fuertes, con salidas estructuradas
* [Eliza](https://github.com/ai16z/eliza)
* [Spring AI (Alibaba)](https://github.com/alibaba/spring-ai-alibaba): Para Java

#### Low code / No code (interfaz visual o configuración mínima)
* [Flowise](https://github.com/FlowiseAI/Flowise): Constructor visual drag-and-drop de agentes y workflows LLM, basado en LangChain, muy popular para prototipos y MVPs sin código
* [Langflow](https://www.langflow.org/) ([GitHub](https://github.com/langflow-ai/langflow)): Visual, similar a Flowise pero más orientado a flujos complejos
* [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio): IDE visual para construir y debuggear grafos de LangGraph
* [Pyspur](https://github.com/PySpur-Dev/pyspur): Intuitivo con muchas características
* [n8n](https://github.com/n8n-io/n8n): Plataforma de automatización con soporte nativo para agentes LLM, 400+ conectores, muy popular (190k+ estrellas), permite automatizaciones complejas con o sin código
* [Dify](https://github.com/langgenius/dify): Framework todo en uno para fácil deploy, con UI de administración para prompts, usuarios y base de conocimiento
* [Tensorzero](https://github.com/tensorzero/tensorzero): Framework todo en uno para fácil deploy
* [Crew AI GUI QT (LangGraph)](https://github.com/LangGraph-GUI/CrewAI-GUI-Qt): Interfaz de escritorio para construir agentes de crewai

#### Memoria para agentes
* [Agno](https://github.com/agno-agi/agno): Incluye agentes, memoria, tool calling, es fácil de usar
* [Mem0](https://github.com/mem0ai/mem0): Sistema de memoria, de lo mejor y más completo para memoria en LLM

### 🗄️ Bases de Datos de Embeddings y Visualizadores

Herramientas para almacenar, buscar y visualizar embeddings/vectores, fundamentales para RAG y búsqueda semántica.

* [ChromaDB](https://github.com/chroma-core/chroma): La más fácil de usar, ideal para prototipos y desarrollo local, open source y Python-native
* [Qdrant](https://github.com/qdrant/qdrant): Escrita en Rust, muy rápida y eficiente, excelente filtrado con metadatos, recomendada para producción con recursos limitados
* [Weaviate](https://github.com/weaviate/weaviate): Open source, soporta búsqueda híbrida (vector + keyword) de forma nativa, buen multi-tenancy
* [Milvus](https://github.com/milvus-io/milvus): Para escala masiva (miles de millones de vectores), más complejo de desplegar pero muy potente
* [FAISS (Facebook)](https://github.com/facebookresearch/faiss): Librería de bajo nivel para búsqueda de vectores, muy rápida en CPU/GPU, embebida en tu código
* [Pinecone](https://www.pinecone.io/): Solo como servicio gestionado (no open source), muy fácil de escalar, buena opción si no quieres gestionar infraestructura
* [LanceDB](https://github.com/lancedb/lancedb): Serverless, orientada a ML/AI, con soporte para columnas multimodales
* [Embedding Projector (TensorFlow)](https://projector.tensorflow.org/): Visualizador interactivo de embeddings, permite ver clusters y relaciones en 3D/2D (PCA, t-SNE, UMAP)
* [Nomic Atlas](https://github.com/nomic-ai/nomic): Plataforma para explorar y visualizar grandes conjuntos de embeddings de texto/imágenes

### 🔭 Herramientas de Observabilidad
* [Opik](https://github.com/comet-ml/opik): Incluye dashboard
* [Phoenix](https://github.com/Arize-ai/phoenix): Incluye dashboard
* [Openllmetry](https://github.com/traceloop/openllmetry): Emite trazas de llm en formato opentelemetry, lo cual le permite conectarse a casi cualquier herramienta de observabilidad de logs o de monitoreo, como kibana, grafana, openobserve, entre otros, tiene muchas integraciones de modelos
* [Agentops](https://github.com/AgentOps-AI/agentops): Se integra con frameworks de agentes (CrewAI, CamelAI, Autogen, LangGraph), tiene más de mil evaluaciones, y guarda los flujos y tiene dashboard
* [Langwatch](https://github.com/langwatch/langwatch): Incluye Dashboard, y montón de herramientas como human in the loop, evaluación, datasets, trazas y workflows, etc
* [DeepEval](https://github.com/confident-ai/deepeval): Incluye dashboard, centrado en la evaluación y benchmarks, no tanto en observabilidad y monitoreo, aunque también se puede usar con ese fin
* [Dify](https://github.com/langgenius/dify): Framework todo en uno para fácil deploy
* [Tensorzero](https://github.com/tensorzero/tensorzero): Framework todo en uno para fácil deploy

### 📄 Preprocesamiento (de PDF u otro a Markdown/JSON)
* [OlmoOCR](https://github.com/allenai/olmocr): Pasa pdf, png, jpeg a Markdown, basado en un VLM de 7B finetuneado, requiere una GPU con 20G de memoria (al menos)
* [Markitdown (Microsoft)](https://github.com/microsoft/markitdown): Herramienta para pasar de Office, Youtube, EPUB y PDF a Markdown
* [Docling (IBM)](https://github.com/docling-project/docling): Herramienta que se puede usar as-is y usa una heurística para pasar de PDF y otros a markdown/json/html/txt, o usar modelos VLM, desde smoldocling de 360M hasta Granite de 3.3B y 7B, especializados en procesamiento de pdfs
* [Marker](https://github.com/datalab-to/marker): Herramienta bastante potente que usa varios modelos "pequeños" para diferentes tareas, haciéndolo más rápido que otras opciones con VLM y pide menos memoria. transforma office, epub y pdf a markdown, también puede usar un llm para corregir el resultado, con varios proveedores disponibles (gemini, ollama, openai, etc)
* [MinerU](https://github.com/opendatalab/MinerU): Requiere mínimo 16GB recomendado 32GB, cpu o gpu o mps
* [MegaParser](https://github.com/QuivrHQ/MegaParse): Usa api de openai o anthropic, y tesseract para ocr, lo que lo hace liviano de ejecutarse pero necesitas tener la api.

## 📦 VectorStores Precomputados
### Word Vectors
* [Word Vector 3B Words](https://www.kaggle.com/datasets/julianusugaortiz/spanish-3b-words-word2vec-embeddings)
* [Word vector 1B words](https://www.kaggle.com/datasets/rtatman/pretrained-word-vectors-for-spanish)
* [DCC Uchile](https://github.com/dccuchile/spanish-word-embeddings)

### Sentence Vectors
* [Spanish Sentence Embeddings](https://github.com/BotCenter/spanish-sent2vec): Calculadas usando el dataset SUC y el programa [sent2vec](https://github.com/epfml/sent2vec)

## 📏 Evaluación y Métricas
* [Wayra Perplexity Estimator](https://huggingface.co/latam-gpt/Wayra-Perplexity-Estimator-55M): 55M, modelo compacto para estimar perplejidad en textos en español, útil para evaluar modelos de lenguaje

### Benchmarks
* [CHOCLO](https://huggingface.co/datasets/latam-gpt/CHOCLO): Benchmark de conocimiento cultural latinoamericano con 100K+ preguntas sobre geografía, fauna, flora, gastronomía y cultura de 18 países de América Latina, en tres niveles de dificultad. Creado por CENIA/Latam-GPT. Licencia MIT
* [TRUEQUE](https://huggingface.co/datasets/latam-gpt/Trueque-Benchmark-beta-0.1): Benchmark colaborativo revisado por humanos, con 500 preguntas sobre historia, cultura, geografía y gastronomía de 20 países de América Latina. Disponible en español y portugués. Creado por CENIA/Latam-GPT. Licencia Apache 2.0
* [IberBench](https://huggingface.co/iberbench): Benchmark multilingüe y multitarea para evaluar LLMs en lenguas ibéricas (español de España y LATAM, portugués, catalán, euskera, gallego), con leaderboard público. [Ver leaderboard](https://huggingface.co/spaces/iberbench/leaderboard)

## 📚 Datasets
* [Colección de datasets en español por metatext](https://metatext.io/datasets-list/spanish-language)
* [Bertin Project](https://huggingface.co/bertin-project): Proyecto para crear GPT y Bert en español. Tienen datasets de tipo Alpaca y similar (pares pregunta - respuesta para crear modelos Instruct)
* [Red Pajama](https://huggingface.co/datasets/latam-gpt/red_pajama_es_hq): Dataset ENORME en español de texto no estructurado para entrenar un LLM, con un score de 2 a 5 según su calidad académica


### No Etiquetado
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

### Etiquetado
* [Datasets QA en HF](https://huggingface.co/datasets?task_categories=task_categories:question-answering&language=language:es&sort=trending)
* [HEAR - Hispanic Emotional Accompaniment Responses](https://huggingface.co/datasets/BrunoGR/HEAR-Hispanic_Emotional_Accompaniment_Responses?row=2): Se usaron datos sintéticos para balancear dataset. par pregunta-respuesta para acompañamiento emocional
* [Punta Cana reviews](https://huggingface.co/datasets/beltrewilton/punta-cana-spanish-reviews): útil para sistemas recomendadores u otras tareas
* [Reviews de hoteles de Andalucía](https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced)
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

---

## 🗂️ Mis Datasets (erickfmm)

Datasets creados o recopilados por el autor de este repositorio, disponibles en HuggingFace y Kaggle.

### HuggingFace ([perfil erickfmm](https://huggingface.co/erickfmm))
* [embedding-data__QQP_triplets__translated_to_spanish](https://huggingface.co/datasets/erickfmm/embedding-data__QQP_triplets__translated_to_spanish): Triplets QQP traducidos al español para entrenamiento de embeddings
* [embedding-data__PAQ_pairs__translated_to_spanish](https://huggingface.co/datasets/erickfmm/embedding-data__PAQ_pairs__translated_to_spanish): Pares PAQ traducidos al español para embeddings
* [wiktionary-spanish-all-forms-and-its-original-form](https://huggingface.co/datasets/erickfmm/wiktionary-spanish-all-forms-and-its-original-form): Todas las formas del Wiktionary en español y sus formas originales
* [wiktionary-spanish-all-meaning-of-original-forms](https://huggingface.co/datasets/erickfmm/wiktionary-spanish-all-meaning-of-original-forms): Significados de las formas originales en español según el Wiktionary
* [agentlans__multilingual-sentences__paired_10_sts](https://huggingface.co/datasets/erickfmm/agentlans__multilingual-sentences__paired_10_sts): Pares de oraciones multilingüe con score STS, útil para entrenamiento de modelos de similitud
* [Tesis_UChile](https://huggingface.co/datasets/erickfmm/Tesis_UChile): Corpus de tesis de la Universidad de Chile
* [filtered_k12_resample_no_chinese](https://huggingface.co/datasets/erickfmm/filtered_k12_resample_no_chinese): Dataset de educación K-12 filtrado y resampleado, sin chino
* [red_pajama_es_hq_45](https://huggingface.co/datasets/erickfmm/red_pajama_es_hq_45): Subconjunto de Red Pajama en español con score de calidad >= 4.5
* [red_pajama_es_hq_35](https://huggingface.co/datasets/erickfmm/red_pajama_es_hq_35): Subconjunto de Red Pajama en español con score de calidad >= 3.5
* [OpenO1-SFT-ESP_10000first_](https://huggingface.co/datasets/erickfmm/OpenO1-SFT-ESP_10000first_): Primeros 10000 ejemplos de OpenO1-SFT traducidos al español, para SFT de razonamiento
* [OpenLongCoT-Pretrain-ES](https://huggingface.co/datasets/erickfmm/OpenLongCoT-Pretrain-ES): Dataset de cadenas de pensamiento largas (Long CoT) traducidas al español para preentrenamiento

### Kaggle ([perfil erickfmm](https://www.kaggle.com/erickfmm))
Datos obtenidos mediante scraping a datos públicos MINEDUC (Ministerio de Educación de Chile):
* [Proyectos Educativos Institucionales (PEI)](https://www.kaggle.com/datasets/erickfmm/education-pei-pdf): En pdf
* [Reglamentos de Convivencia](https://www.kaggle.com/datasets/erickfmm/education-reglamento-convivencia-pdf): En pdf
* [Reglamentos de Evaluación](https://www.kaggle.com/datasets/erickfmm/education-reglamentos-de-evaluacin-pdf): En pdf
