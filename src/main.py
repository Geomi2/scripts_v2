import os
from utils.populate_db_v3 import load_pdf,split_pdfs,process_pdf
from utils.summaries_v3 import check_chroma_integrity,book_choice,generate_summary
from settings import BOOKS_PATH,BOOK_TITLE


class ProcessClient:
    def _populate_datatbase(book_path):
        try:
            for file_name in os.listdir(book_path):
                if file_name.endswith(".pdf"):
                    books = load_pdf(file_name)
                    chunks = split_pdfs(books)
                    process_pdf(file_name, chunks)
            return True
        except Exception as e:
            return False
    
    def _create_summary(book):
        check_chroma_integrity()
        book_for_summary = book_choice(book)
        generate_summary(book_for_summary)


def main():
    ProcessClient._populate_datatbase(BOOKS_PATH)
    ProcessClient._create_summary(BOOK_TITLE)

if __name__ == "__main__":
    main()