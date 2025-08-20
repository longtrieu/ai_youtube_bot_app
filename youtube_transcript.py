import re
from youtube_transcript_api import YouTubeTranscriptApi

def get_video_id(url):
  """Extract video ID from YouTube URL using regex pattern."""
  pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
  match = re.search(pattern, url)
  return match.group(1) if match else None

def get_transcript(url):
  """Extract transcript from YouTube video URL."""
  video_id = get_video_id(url)

  if not video_id:
    return None

  ytt_api = YouTubeTranscriptApi()
  transcripts = ytt_api.list(video_id)

  transcript = ""
  for t in transcripts:
    if t.language_code == 'en':
      if t.is_generated:
        if len(transcript) == 0:
          transcript = t.fetch()
      else:
        transcript = t.fetch()
        break

  return transcript if transcript else None

def process(transcript):
  """Process transcript into formatted text."""
  txt = ""
  for i in transcript:
    try:
      txt += f"Text: {i.text} Start: {i.start}\n"
    except KeyError:
      pass
  return txt
