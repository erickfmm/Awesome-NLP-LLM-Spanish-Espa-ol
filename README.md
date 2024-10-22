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


### LLM multi idioma (incluye español)
* [Meta Llama 3.2](https://huggingface.co/collections/unsloth/llama-32-all-versions-66f46afde4ca573864321a22): Versiones 1B, 3B, 11B(Vision), 90B
* [Meta LLama 3.1](https://llama.meta.com/): Versiones 8B, 70B, 405B
* [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
* [Ministral 8B](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)
* Mixtral 8x7B
* [Mistral Large (123B)](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407)
* [Mistral Nemo (12B)](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
* [Mixtral 8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)
* [Phi 3.5 mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
* [Phi 3.5 MoE](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)
* [Phi 3.5 Vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
* [Molmo](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)
* [EuroLLM](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct): Por utter project, consorcio de varias universidades de Alemania
* [Apollo](https://huggingface.co/collections/FreedomIntelligence/apollomoe-and-apollo2-670ddebe3bb1ba1aebabbf2c): Profesor de una universidad China

## Modelos BERT, BART y Transformers en Español
* Departamento de Ciencias de la Computación Universidad de Chile (DCC Uchile)
  * [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased): Modelo BERT Finetuneado de BERT-base (original de Google) con 768 dimensiones, entrenado en MASKED LANGUAGE y finetuneado en varias tareas más
  * [Patana](https://huggingface.co/dccuchile/patana-chilean-spanish-bert): Finetunning de BETO con texto de Chile
  * [Tulio](https://huggingface.co/dccuchile/tulio-chilean-spanish-bert): Finetuning de BETO con texto de Chile y libros en Español.
  * [Versiones livianas](https://github.com/dccuchile/lightweight-spanish-language-models)
* [NV-Embed](https://huggingface.co/nvidia/NV-Embed-v1): Generador de Embeddings basado en un LLM Mistral (multi idioma)
* [BETO finetuned on XNLI](https://huggingface.co/Recognai/bert-base-spanish-wwm-cased-xnli)
* [BETO FINETUNED en Sentence Similarity](https://huggingface.co/hiiamsid/sentence_similarity_spanish_es)
* [BETO finetuned en summarization](https://huggingface.co/mrm8488/bert2bert_shared-spanish-finetuned-summarization)
* [Bertin Project](https://huggingface.co/bertin-project): Proyecto con datasets para crear un BERT en español. y modelos GPT.
* [JinaAI](https://huggingface.co/jinaai/jina-embeddings-v3): Multi idioma, multi tarea query-retrieval
* [LaBSE](https://huggingface.co/sentence-transformers/LaBSE): Multi idioma

## Named Entity Recognition
* [Wikineural](https://huggingface.co/Babelscape/wikineural-multilingual-ner)
* [Detección de información personal por token](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information)

## Traducción
* nllb-200 (Facebook): Versiones [1.3B](https://huggingface.co/facebook/nllb-200-1.3B) y [3.3B](https://huggingface.co/facebook/nllb-200-3.3B) y versiones distilled
* [Seamless (Facebook)](https://huggingface.co/facebook/seamless-m4t-v2-large): Audio a texto, texto a texto

## Speech Recognition
* [Whisper (v3 large turbo)](https://huggingface.co/openai/whisper-large-v3-turbo): Modelo de OpenAI, versiones [small](https://huggingface.co/openai/whisper-small), [tiny](https://huggingface.co/openai/whisper-tiny)
* [Canary (Nvidia)](https://huggingface.co/nvidia/canary-1b)

## Texto To Speech
* [MMS (Facebook)](https://huggingface.co/facebook/mms-tts-spa): Español
* [Bark (Suno)](https://huggingface.co/suno/bark): Multiidioma

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
