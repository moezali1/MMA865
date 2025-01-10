from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from engine import BlogGenerator
from database import connect_to_database
from bson.json_util import dumps
import json
import os

app = FastAPI()
db = connect_to_database()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/create_blog")
async def create_blog(topic: str):
    # Initialize blog generator
    blog_generator = BlogGenerator(topic)

    # Generate blog
    result = blog_generator.create_blog()

    # Store blog content in MongoDB if generation was successful
    if result:
        try:
            # Insert blog content into blogs collection
            inserted_doc = db.blogs.insert_one(blog_generator.blog_content)

            # Save HTML file
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"blog_{inserted_doc.inserted_id}.html")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(blog_generator.html_output)

            # Convert MongoDB document to JSON-serializable format
            blog_content = json.loads(dumps(blog_generator.blog_content))

            return {
                "message": "Blog created successfully",
                "blog_id": str(inserted_doc.inserted_id),
                "content": blog_content,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error saving to database: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating blog: {result.get('error', 'Unknown error')}",
        )


@app.get("/get_blogs")
async def get_blogs():
    try:
        # Get all blogs from database
        blogs = db.blogs.find({}, {"content": 0})  # Exclude content field

        # Convert MongoDB cursor to JSON-serializable format
        blogs_list = json.loads(dumps(blogs))

        return {"message": "Blogs retrieved successfully", "blogs": blogs_list}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving blogs from database: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
