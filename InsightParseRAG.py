import os
import base64
from huggingface_hub import login
from byaldi import RAGMultiModalModel
from pdf2image import convert_from_path
from IPython.display import Image, display

# 1. SECURE AUTHENTICATION
login(token="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") 

# 2. INITIALIZE VISION ENGINE
print("Initializing ColPali v1.2 Engine...")
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

# 3. DOCUMENT INGESTION PIPELINE
input_folder = "/media/my_docs"
processed_folder = "processed_visuals"

if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

print(f"Reading documents from: {input_folder}")
pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]

if not pdf_files:
    print("ERROR: No PDFs found! Double check the files in /media/my_docs")
else:
    for pdf in pdf_files:
        print(f"Converting {pdf} to visual patches...")
        pages = convert_from_path(os.path.join(input_folder, pdf))
        for i, page in enumerate(pages):
            page.save(f"{processed_folder}/{pdf}_page_{i}.png", "PNG")

    # 4. BUILDING THE VISUAL INDEX
    print(f"Building Visual Index for {len(os.listdir(processed_folder))} pages...")
    RAG.index(
        input_path=processed_folder,
        index_name="insight_parse_final",
        store_collection_with_index=True,
        overwrite=True
    )

    # 5. THE SEARCH QUERY
    query = "Find a page with a complex data table, financial chart, or technical diagram."
    results = RAG.search(query, k=1)

    # 6. GENERATE FINAL OUTPUT
    image_data = base64.b64decode(results[0].base64)
    with open("final_submission_result.jpg", "wb") as f:
        f.write(image_data)

    print("\n--- PROJECT SUCCESS: INSIGHTPARSE ENGINE IS LIVE ---")
    display(Image("final_submission_result.jpg"))