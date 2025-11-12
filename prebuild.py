import os
import sys
from RagFullPipeline import VectorStore
import time


def prebuild_vector_store():
    print("🚀 Pre-building vector store...")
    start_time = time.time()

    try:
        vectorstore = VectorStore(
            persist_directory="./prebuilt_vector_store",
            use_persistent=True,
            pdf_folder="."  # Current directory
        )

        doc_count = vectorstore.collection.count()
        end_time = time.time()

        print(f"✅ Vector store built successfully!")
        print(f"📚 Documents: {doc_count}")
        print(f"⏱️ Time taken: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Error building vector store: {e}")
        sys.exit(1)


if __name__ == "__main__":
    prebuild_vector_store()