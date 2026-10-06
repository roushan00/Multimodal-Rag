"""python -m scripts.ingest data/pdfs/*.pdf"""
import json
import sys

from app.ingest.pipeline import ingest_pdf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m scripts.ingest <pdf> [<pdf> ...]")
    for path in sys.argv[1:]:
        print(json.dumps(ingest_pdf(path)))
