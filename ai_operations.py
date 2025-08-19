from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def create_summary_prompt():
  """Create a PromptTemplate for summarizing a YouTube video transcript."""
  template = """
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

  Instructions:
  1. Summarize the transcript in a single concise paragraph.
  2. Ignore any timestamps in your summary.
  3. Focus on the spoken content (Text) of the video.

  Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
  Please summarize the following YouTube video transcript:

  {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  """

  prompt = PromptTemplate(
    input_variables=["transcript"],
    template=template
  )
  return prompt

def create_summary_chain(llm, prompt, verbose=True):
  """Create an LLMChain for generating summaries."""
  return LLMChain(llm=llm, prompt=prompt, verbose=verbose)

def create_qa_prompt_template():
  """Create a PromptTemplate for question answering based on video content."""
  qa_template = """
  You are an expert assistant providing detailed answers based on the following video content.
  Relevant Video Context: {context}
  Based on the above context, please answer the following question:
  Question: {question}
  """
  prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=qa_template
  )
  return prompt_template

def create_qa_chain(llm, prompt_template, verbose=True):
  """Create an LLMChain for question answering."""
  return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)

def generate_answer(question, faiss_index, qa_chain, k=7):
  """Retrieve relevant context and generate an answer based on user input."""
  from vector_operations import retrieve

  relevant_context = retrieve(question, faiss_index, k=k)
  answer = qa_chain.predict(context=relevant_context, question=question)
  return answer
