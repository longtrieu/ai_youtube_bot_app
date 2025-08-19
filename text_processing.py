from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
  """Split transcript into manageable chunks for processing."""
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
  )
  chunks = text_splitter.split_text(processed_transcript)
  return chunks
