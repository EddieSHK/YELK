from pptx import Presentation
from transformers import pipeline

#Reading Slides
def extract_text_from_pptx(file_path):
    prs = Presentation(file_path)
    slide_texts = []
    for slide in prs.slides:
        slide_content = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                slide_content.append(shape.text)
        slide_texts.append("\n".join(slide_content))
    return slide_texts

# Example usage:
slides = extract_text_from_pptx("example.pptx")
# print(slides)

# Summarizing Text
def summarize_text(text):
    summarizer = pipeline("summarization")
    return summarizer(text, max_length=3, min_length=1, do_sample=False)[0]["summary_text"]

# Example usage:
summary = summarize_text("slide")
print(summary)
