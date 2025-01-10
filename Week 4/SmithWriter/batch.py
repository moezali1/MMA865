import time
from engine import BlogGenerator
from database import connect_to_database
import os


def process_topics():
    db = connect_to_database()

    while True:
        try:
            # Read topics from input.txt
            with open("input.txt", "r") as file:
                topics = file.readlines()

            # Process each topic
            for topic in topics:
                topic = topic.strip()  # Remove whitespace and newlines
                if not topic:  # Skip empty lines
                    continue

                print(f"Processing topic: {topic}")

                # Initialize blog generator
                blog_generator = BlogGenerator(topic)

                # Generate blog
                result = blog_generator.create_blog()

                # Store blog content in MongoDB if generation was successful
                if result:
                    try:
                        inserted_doc = db.blogs.insert_one(blog_generator.blog_content)
                        print(
                            f"Blog content saved to database with id: {inserted_doc.inserted_id}"
                        )

                        # Save HTML file
                        output_dir = os.path.join(os.getcwd(), "output")
                        os.makedirs(output_dir, exist_ok=True)
                        filepath = os.path.join(
                            output_dir, f"blog_{inserted_doc.inserted_id}.html"
                        )

                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(blog_generator.html_output)
                        print(f"HTML file saved at: {filepath}")

                        # Remove processed topic from input.txt
                        with open("input.txt", "w") as file:
                            file.writelines(
                                [line for line in topics if line.strip() != topic]
                            )

                    except Exception as e:
                        print(f"Error saving to database: {str(e)}")
                else:
                    print(
                        "Error generating blog:", result.get("error", "Unknown error")
                    )
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")


if __name__ == "__main__":
    process_topics()
