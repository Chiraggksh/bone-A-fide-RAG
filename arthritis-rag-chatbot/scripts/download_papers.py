import arxiv
import os
import time
import pandas as pd
import requests

from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================

TOPIC_QUERIES = [

    # Arthritis
    'all:"arthritis"',
    'all:"osteoarthritis"',
    'all:"rheumatoid arthritis"',
    'all:"juvenile arthritis"',
    'all:"psoriatic arthritis"',
    'all:"inflammatory arthritis"',
    'all:"chronic arthritis"',

    # Bone diseases
    'all:"osteoporosis"',
    'all:"bone disease"',
    'all:"bone degeneration"',
    'all:"bone loss"',
    'all:"bone density"',

    # Bone repair
    'all:"bone fracture"',
    'all:"bone regeneration"',
    'all:"bone healing"',
    'all:"bone remodeling"',
    'all:"bone tissue engineering"',

    # Joint related
    'all:"joint disease"',
    'all:"joint inflammation"',
    'all:"joint degeneration"',
    'all:"cartilage damage"',
    'all:"cartilage degeneration"',

    # Medical specialties
    'all:"orthopedic"',
    'all:"orthopaedic"',
    'all:"skeletal disease"',
    'all:"musculoskeletal"',
    'all:"rheumatology"',

    # Imaging & diagnosis
    'all:"bone imaging"',
    'all:"arthritis diagnosis"',
    'all:"medical bone diagnosis"',
    'all:"xray arthritis"',
    'all:"mri bone"'
]

# Target dataset size
MAX_TOTAL_PAPERS = 3000

# Per query fetch size
RESULTS_PER_QUERY = 250

# Delay to avoid HTTP 429
REQUEST_DELAY = 10

PDF_DIR = "data/raw_pdfs"
METADATA_PATH = "data/metadata/papers_metadata.csv"

# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)

# ==========================================
# LOAD EXISTING PAPERS
# ==========================================

existing_papers = set()

for filename in os.listdir(PDF_DIR):

    if filename.endswith(".pdf"):

        paper_id = filename.replace(
            ".pdf",
            ""
        )

        existing_papers.add(
            paper_id
        )

print(
    f"Already downloaded papers: "
    f"{len(existing_papers)}"
)

# ==========================================
# SEARCH PAPERS
# ==========================================

print("\nSearching relevant arXiv papers...")

client = arxiv.Client()

all_results = []

for query in TOPIC_QUERIES:

    print(f"\nSearching: {query}")

    try:

        search = arxiv.Search(
            query=query,
            max_results=RESULTS_PER_QUERY,
            sort_by=arxiv.SortCriterion.Relevance
        )

        query_results = []

        for result in client.results(search):

            query_results.append(result)

            # stop early if enough total
            if len(all_results) >= (
                MAX_TOTAL_PAPERS * 2
            ):
                break

        print(
            f"Found "
            f"{len(query_results)} papers"
        )

        all_results.extend(
            query_results
        )

        print(
            f"Current total raw papers: "
            f"{len(all_results)}"
        )

        print(
            f"Waiting "
            f"{REQUEST_DELAY}s..."
        )

        time.sleep(
            REQUEST_DELAY
        )

    except Exception as e:

        print(
            f"Search failed "
            f"for {query}: {e}"
        )
# ==========================================
# REMOVE DUPLICATES
# ==========================================

unique_results = {}

for paper in all_results:

    paper_id = (
        paper.entry_id
        .split("/")[-1]
    )

    unique_results[
        paper_id
    ] = paper

results = list(
    unique_results.values()
)

print(
    f"\nFound "
    f"{len(results)} "
    f"unique papers"
)

# ==========================================
# RELEVANCE FILTER
# ==========================================

relevant_keywords = [

    "arthritis",
    "osteoarthritis",
    "rheumatoid",
    "psoriatic",
    "juvenile arthritis",

    "osteoporosis",
    "bone",
    "skeletal",
    "fracture",
    "bone regeneration",
    "bone remodeling",
    "bone density",

    "orthopedic",
    "orthopaedic",

    "joint",
    "cartilage",

    "musculoskeletal",
    "rheumatology"
]

filtered_results = []

for paper in results:

    text_to_check = (
        paper.title.lower()
        + " "
        + paper.summary.lower()
    )

    if any(
        keyword in text_to_check
        for keyword in relevant_keywords
    ):
        filtered_results.append(
            paper
        )

print(
    f"Relevant papers "
    f"after filtering: "
    f"{len(filtered_results)}"
)

# ==========================================
# LIMIT TO TARGET COUNT
# ==========================================

filtered_results = (
    filtered_results[
        :MAX_TOTAL_PAPERS
    ]
)

print(
    f"Target papers: "
    f"{len(filtered_results)}"
)

# ==========================================
# DOWNLOAD PAPERS
# ==========================================

metadata = []

downloaded_count = 0
skipped_count = 0
failed_count = 0

print("\nDownloading PDFs...\n")

for paper in tqdm(
    filtered_results,
    desc="Downloading papers"
):

    try:

        paper_id = (
            paper.entry_id
            .split("/")[-1]
        )

        pdf_filename = (
            f"{paper_id}.pdf"
        )

        pdf_path = os.path.join(
            PDF_DIR,
            pdf_filename
        )

        # ======================
        # SKIP IF EXISTS
        # ======================

        if (
            paper_id
            in existing_papers
        ):

            skipped_count += 1

            continue

        # ======================
        # DOWNLOAD PDF
        # ======================

        response = requests.get(
            paper.pdf_url,
            timeout=30
        )

        if (
            response.status_code
            == 200
        ):

            with open(
                pdf_path,
                "wb"
            ) as f:

                f.write(
                    response.content
                )

            downloaded_count += 1

        else:

            failed_count += 1

            continue

        # ======================
        # SAVE METADATA
        # ======================

        metadata.append(
            {
                "paper_id":
                    paper_id,

                "title":
                    paper.title,

                "authors":
                    ", ".join(
                        [
                            author.name
                            for author
                            in paper.authors
                        ]
                    ),

                "summary":
                    paper.summary,

                "published":
                    paper.published,

                "pdf_path":
                    pdf_path,

                "pdf_url":
                    paper.pdf_url,

                "categories":
                    ", ".join(
                        paper.categories
                    )
            }
        )

        time.sleep(0.5)

    except Exception as e:

        failed_count += 1

        print(
            f"\nError "
            f"downloading "
            f"{paper_id}: {e}"
        )

# ==========================================
# SAVE METADATA
# ==========================================

df = pd.DataFrame(metadata)

if os.path.exists(
    METADATA_PATH
):

    old_df = pd.read_csv(
        METADATA_PATH
    )

    df = pd.concat(
        [old_df, df],
        ignore_index=True
    )

    df.drop_duplicates(
        subset="paper_id",
        inplace=True
    )

df.to_csv(
    METADATA_PATH,
    index=False
)

# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 50)

print("DOWNLOAD COMPLETE")

print("=" * 50)

print(
    f"New downloads: "
    f"{downloaded_count}"
)

print(
    f"Skipped "
    f"(already exists): "
    f"{skipped_count}"
)

print(
    f"Failed: "
    f"{failed_count}"
)

print(
    f"Total PDFs now: "
    f"{len(os.listdir(PDF_DIR))}"
)

print(
    f"\nPapers saved to: "
    f"{PDF_DIR}"
)

print(
    f"Metadata saved to: "
    f"{METADATA_PATH}"
)