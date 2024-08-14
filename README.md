# Awesome-NLP-LLM-Spanish-Espa-ol
Listado de recursos, datasets, modelos, cursos y otros relacionados al NLP y LLM en Español

## Otros Awesome similares
* [LLMs In Spanish](https://github.com/drcaiomoreno/LLMsInSpanish)
* [Lacuna Fund](https://lacunafund.org/language-resources/): Recursos varios

## LLM en español
* [LINCE ZERO](https://huggingface.co/clibrain/lince-zero): Modelo español simple (ZERO)
* Projecte AINA (Español y catalán)
  * [Flor 1.3B-Instructed](https://huggingface.co/projecte-aina/FLOR-1.3B-Instructed)
  * [Flor 6.3B](https://huggingface.co/projecte-aina/FLOR-6.3B)
  * [Aguila 7B](https://huggingface.co/projecte-aina/aguila-7b): Basado en Falcon7B

### LLM multi idioma (incluye español)
* [Meta LLama 3.1](): Versiones 8B, 12B, 400B
* Mistral
* Mixtral 8x7B
* Mixtral 8x22B

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
* [Gensim](): Para hacer LDA (Análisis de tópicos) y otras tareas.
* pyLDAVis: Para visualizar los tópicos LDA generados con gensim
* BERTopics: Para generar tópicos usando BERT

## VectorStores
### Word Vectors
* [Word Vector 3B Words](https://www.kaggle.com/datasets/julianusugaortiz/spanish-3b-words-word2vec-embeddings)
* [Word vector 1B words](https://www.kaggle.com/datasets/rtatman/pretrained-word-vectors-for-spanish)
* [DCC Uchile](https://github.com/dccuchile/spanish-word-embeddings)

### Sentence Vectors

## Datasets
### No etiquetado
* DCC Uchile
  * [Spanish Unanotated Corpora](https://github.com/josecannete/spanish-corpora): Usado para entrenar BETO
  * [Spanish Books](https://huggingface.co/datasets/jorgeortizfuentes/spanish_books)
  * [Chilean Spanish Corpus](https://huggingface.co/datasets/jorgeortizfuentes/chilean-spanish-corpus)
  * [Universal Spanish Chilean Corpus](https://huggingface.co/datasets/jorgeortizfuentes/universal_spanish_chilean_corpus): La suma de los dos anteriores (aparentemente)
* [Wikihow en español](https://huggingface.co/datasets/daqc/wikihow-spanish)
* [Wikipedia as TXT](https://huggingface.co/datasets/daqc/wikipedia-txt-spanish)
* [Wiktionary](https://huggingface.co/datasets/carloscapote/es.wiktionary.org): Diccionario de la Wikipedia en español
* [9322 letras de rap en español](https://www.kaggle.com/datasets/smunoz3801/9325-letras-de-rap-en-espaol/data)
* [Wikibooks (Multi idioma)](https://www.kaggle.com/datasets/dhruvildave/wikibooks-dataset)
* Obtenidos mediante Scraping a datos públicos MINEDUC (Ministerio de Educación de Chile)
  * [Proyectos Educativos Institucionales (PEI)](https://www.kaggle.com/datasets/erickfmm/education-pei-pdf): En pdf
  * [Reglamentos de Convivencia](https://www.kaggle.com/datasets/erickfmm/education-reglamento-convivencia-pdf): En pdf
  * [Reglamentos de Evaluación](https://www.kaggle.com/datasets/erickfmm/education-reglamentos-de-evaluacin-pdf): En pdf

### Etiquetado
* [HEAR - Hispanic Emotional Accompaniment Responses](https://huggingface.co/datasets/BrunoGR/HEAR-Hispanic_Emotional_Accompaniment_Responses?row=2): Se usaron datos sintéticos para balancear dataset. par pregunta-respuesta para acompañamiento emocional
* [Punta Cana reviews](https://huggingface.co/datasets/beltrewilton/punta-cana-spanish-reviews): útil para sistemas recomendadores u otras tareas
* [Reviews de hoteles de Andalucia](https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced)
* [Reviews de comida peruana](https://www.kaggle.com/datasets/lazaro97/peruvian-food-reviews)
* [Reseñas IMDB TRADUCIDAS al español](https://www.kaggle.com/datasets/luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish)
* [XNLI (Inferencia dada una premisa y hipótesis), Multiidioma](https://huggingface.co/datasets/facebook/xnli)
* [HateSpeech (Multi idioma)](https://www.kaggle.com/datasets/wajidhassanmoosa/multilingual-hatespeech-dataset)
* [STS (Sentence Similarity) Traducido con DeepL a varios idiomas](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt)
* [DWUG ES: Diachronic Word Usage Graphs for Spanish](https://zenodo.org/records/6433667#.YmGU7i8lP0o)
