"""See what the parser extracts before anything is embedded: python -m scripts.inspect_pdf file.pdf"""
import sys

from app.ingest.pdf_parser import parse_pdf, to_chunks

if __name__ == "__main__":
    doc = parse_pdf(sys.argv[1])
    for p in doc.pages:
        chars = sum(len(t) for t in p.text_blocks)
        flag = "  <-- little/no text: scanned page? only Path B will see it" if chars < 50 else ""
        print(f"p.{p.page}: {len(p.text_blocks)} blocks / {chars} chars, "
              f"{len(p.tables_md)} tables, {len(p.figures)} figures{flag}")
        for md in p.tables_md:
            print("   table preview:", md.splitlines()[0][:100])
    chunks = to_chunks(doc)
    print(f"\n{len(chunks)} chunks; renders in data/pages/{doc.doc_id}/")
