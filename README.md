
# Resume Generator with AI

This project demonstrates a resume generator powered by a fine-tuned t5-small transformer model, capable of creating tailored resumes based on specific job descriptions. It uses OCR for extracting text from images and PDFs and employs a transformer-based language model to produce context-specific resumes.

## Features

- **OCR Integration**: Extracts text from job description images and PDFs.
- **Resume Tailoring**: Matches and aligns resumes to specific job requirements using advanced NLP techniques powered by the T5-small model. T5-small, a transformer-based text-to-text model by Google, is fine-tuned to generate text outputs by understanding and mapping job descriptions to relevant resume details
- **End-to-End Workflow**: Combines preprocessing, extraction, and resume generation in a pipeline.

## How It Works

1. **Input Job Description**:
    - OCR extracts job descriptions from provided images or text files.
2. **Extract Resume Data**:
    - Parses and processes original resume content from PDFs.
3. **Generate Tailored Resume**:
    - Uses a pre-trained transformer model to align qualifications and skills with job requirements.
    - Utilizes the T5-small transformer model from the Hugging Face Transformers library. The input prompt combines the job description and the original resume into a structured format.
    - Tokenization is performed using the AutoTokenizer class, converting text to a sequence of input IDs suitable for the model.The generate method of the AutoModelForSeq2SeqLM is used to produce text output. It generates a tailored resume with:
         - A maximum token length of 512.
         - Parameters optimized for single-sequence generation (num_return_sequences=1).
         - Truncation and padding enabled to handle varying input lengths.
   - The output sequence is decoded back into a human-readable format using the tokenizer’s decoding function, ensuring the exclusion of special tokens.

## Usage

### Requirements

- Python 3.8 or higher
- Required libraries:
  ```
  pip install transformers paddleocr PyPDF2
  ```

### Steps

1. Clone the repository:
   ```
   git clone https://github.com/Mhaidar117/Resumama.git
   cd resumama
   ```

2. Run the notebook:
   Open `notebooks/Resume_Generator.ipynb` in Jupyter Notebook and follow the instructions to extract text and generate tailored resumes.

3. Example commands:
   - Extract text from a job description image:
     ```python
     extract_text_with_paddleocr(image_path)
     ```

   - Generate tailored resume:
     ```python
     generate_matching_resume(job_description, resume_text)
     ```

## Example Workflow

1. Input job description image:
   ![Job Description](images/job_description_example.jpg)

2. Processed job description text:
   ```
   Ibotta is seeking a Machine Learning Intern...
   ```
3. Input resume:
   ![Resume](resources/Resume.pdf)
3. Generated tailored resume:
   ```
   Michael Haidar
   Machine Learning Engineer specializing in...
   ```

## Resources

- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [Transformer Models by Hugging Face](https://huggingface.co/docs/transformers/index)
- [Text-to-text transfer transformer (t5)](https://github.com/google-research/text-to-text-transfer-transformer)
- Relevant YouTube Video Tutorials:
  - [Introduction to t5 tuning](https://www.youtube.com/watch?v=PyRbP9d27sk)
  - [Understanding Transformers for NLP](https://www.youtube.com/watch?v=fNxaJsNG3-s)

## Limitations

- Input truncation may lose critical context for longer resumes.
- Current model needs more training to produce resume.


## Future Work

- Enhance dataset diversity for training.
- Expand model capability with additional layers or fine-tuned weights.

---
## Pseduocode:
```
# Step 1: Extract Text from Job Description and Resume

# Input: Job description (image or text file) and resume (PDF file)
job_description_path = "path/to/job_description.jpg"
resume_pdf_path = "path/to/resume.pdf"

# Process Job Description
if job_description_path.endswith(".jpg") or job_description_path.endswith(".png"):
    # Use OCR to extract text from job description image
    from paddleocr import PaddleOCR
    ocr = PaddleOCR()
    job_description_text = " ".join([line[1][0] for line in ocr.ocr(job_description_path)[0]])
else:
    # Read directly from text file
    with open(job_description_path, "r") as file:
        job_description_text = file.read()

# Process Resume
from PyPDF2 import PdfReader
reader = PdfReader(resume_pdf_path)
resume_text = "".join([page.extract_text() for page in reader.pages])

# Step 2: Generate Tailored Resume

# Load Pre-trained Transformer Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Create structured input prompt
input_text = (
    f"Generate a tailored Resume\n\n"
    f"Job Description: {job_description_text}\n\n"
    f"Original Resume: {resume_text}"
)

# Tokenize input
input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids

# Generate tailored resume using the model
output_ids = model.generate(input_ids, max_length=512, num_return_sequences=1)

# Decode output
tailored_resume = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Step 3: Save or Display Tailored Resume
output_path = "path/to/output_tailored_resume.txt"
with open(output_path, "w") as file:
    file.write(tailored_resume)

print("Tailored Resume Generated:", tailored_resume)

```

## Acknowledgments

- Developed as part of the MS Data Science Program at Vanderbilt University.
- [Data from cnamuangtoun](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
