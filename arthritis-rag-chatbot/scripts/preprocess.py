import fitz
import os
from tqdm import tqdm

# ==========================
# PATHS
# ==========================

PDF_DIR = "data/raw_pdfs"
TEXT_DIR = "data/processed_text"

os.makedirs(TEXT_DIR, exist_ok=True)

# ==========================
# GET PDF FILES
# ==========================

pdf_files = [
    file for file in os.listdir(PDF_DIR)
    if file.endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDFs")

# ==========================
# COUNTERS
# ==========================

processed_count = 0
skipped_count = 0
failed_count = 0
empty_count = 0

# ==========================
# PROCESS PDFS
# ==========================

for pdf_file in tqdm(
    pdf_files,
    desc="Extracting text"
):

    try:

        pdf_path = os.path.join(
            PDF_DIR,
            pdf_file
        )

        text_filename = pdf_file.replace(
            ".pdf",
            ".txt"
        )

        text_path = os.path.join(
            TEXT_DIR,
            text_filename
        )

        # ======================
        # SKIP IF ALREADY DONE
        # ======================

        if os.path.exists(
            text_path
        ):

            skipped_count += 1
            continue

        # ======================
        # OPEN PDF
        # ======================

        doc = fitz.open(
            pdf_path
        )

        full_text = ""

        # ======================
        # EXTRACT PAGE TEXT
        # ======================

        for page in doc:

            try:
                page_text = page.get_text()

                if page_text:
                    full_text += (
                        page_text
                        + "\n"
                    )

            except Exception:
                continue

        doc.close()

        # ======================
        # CLEAN TEXT
        # ======================

        full_text = (
            full_text.strip()
        )

        # Skip empty docs
        if len(full_text) < 500:

            empty_count += 1
            continue

        # ======================
        # SAVE TEXT
        # ======================

        with open(
            text_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                full_text
            )

        processed_count += 1

    except Exception as e:

        failed_count += 1

        print(
            f"\nError processing "
            f"{pdf_file}: {e}"
        )

# ==========================
# FINAL REPORT
# ==========================

print("\n" + "=" * 50)
print("TEXT EXTRACTION COMPLETE")
print("=" * 50)

print(
    f"New text files created: "
    f"{processed_count}"
)

print(
    f"Skipped "
    f"(already processed): "
    f"{skipped_count}"
)

print(
    f"Empty/invalid PDFs: "
    f"{empty_count}"
)

print(
    f"Failed: "
    f"{failed_count}"
)

print(
    f"Total text files now: "
    f"{len(os.listdir(TEXT_DIR))}"
)