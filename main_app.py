import gradio as gr

# Import from our modular files
from youtube_transcript import get_transcript, process
from text_processing import chunk_transcript
from watson_config import setup_credentials, define_parameters, initialize_watsonx_llm, setup_embedding_model
from vector_operations import create_faiss_index
from ai_operations import create_summary_prompt, create_summary_chain, create_qa_prompt_template, create_qa_chain, generate_answer

# Global variables for transcript storage
fetched_transcript = None
processed_transcript = ""

def summarize_video(video_url):
  """Generate a summary of the video using the preprocessed transcript."""
  global fetched_transcript, processed_transcript

  if video_url:
    fetched_transcript = get_transcript(video_url)
    processed_transcript = process(fetched_transcript)
  else:
    return "Please provide a valid YouTube URL."

  if processed_transcript:
    # Step 1: Set up IBM Watson credentials
    model_id, credentials, client, project_id = setup_credentials()

    # Step 2: Initialize WatsonX LLM for summarization
    llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

    # Step 3: Create the summary prompt and chain
    summary_prompt = create_summary_prompt()
    summary_chain = create_summary_chain(llm, summary_prompt)

    # Step 4: Generate the video summary
    summary = summary_chain.run({"transcript": processed_transcript})
    return summary
  else:
    return "No transcript available. Please fetch the transcript first."

def answer_question(video_url, user_question):
  """Answer user's question based on video content."""
  global fetched_transcript, processed_transcript

  # Check if the transcript needs to be fetched
  if not processed_transcript:
    if video_url:
      fetched_transcript = get_transcript(video_url)
      processed_transcript = process(fetched_transcript)
    else:
      return "Please provide a valid YouTube URL."

  if processed_transcript and user_question:
    # Step 1: Chunk the transcript (only for Q&A)
    chunks = chunk_transcript(processed_transcript)

    # Step 2: Set up IBM Watson credentials
    model_id, credentials, client, project_id = setup_credentials()

    # Step 3: Initialize WatsonX LLM for Q&A
    llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

    # Step 4: Create FAISS index for transcript chunks (only needed for Q&A)
    embedding_model = setup_embedding_model(credentials, project_id)
    faiss_index = create_faiss_index(chunks, embedding_model)

    # Step 5: Set up the Q&A prompt and chain
    qa_prompt = create_qa_prompt_template()
    qa_chain = create_qa_chain(llm, qa_prompt)

    # Step 6: Generate the answer using FAISS index
    answer = generate_answer(user_question, faiss_index, qa_chain)
    return answer
  else:
    return "Please provide a valid question and ensure the transcript has been fetched."

def create_interface():
  """Create and configure the Gradio interface."""
  with gr.Blocks() as interface:
    # Input field for YouTube URL
    video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter the YouTube Video URL")

    # Outputs for summary and answer
    summary_output = gr.Textbox(label="Video Summary", lines=5)
    question_input = gr.Textbox(label="Ask a Question About the Video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

    # Buttons for selecting functionalities after fetching transcript
    summarize_btn = gr.Button("Summarize Video")
    question_btn = gr.Button("Ask a Question")

    # Display status message for transcript fetch
    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    # Set up button actions
    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

  return interface

if __name__ == "__main__":
  # Create and launch the interface
  interface = create_interface()
  interface.launch(server_name="0.0.0.0", server_port=7860)
