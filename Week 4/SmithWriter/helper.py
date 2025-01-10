import os
from openai import OpenAI
import json
from dotenv import load_dotenv
import re


def convert_to_html(blog_content):
    # Start HTML document with metadata
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{blog_content['metadata']['meta_description']}">
    <meta name="keywords" content="{', '.join(blog_content['metadata']['primary_keywords'] + blog_content['metadata']['secondary_keywords'])}">
    <title>{blog_content['metadata']['meta_title']}</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            font-size: 2em;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            font-size: 1.5em;
            margin-top: 20px;
        }}
        .subtitle {{
            font-size: 1.2em;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .subsection {{
            margin-left: 20px;
            margin-bottom: 20px;
        }}
        .tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .tag {{
            background-color: #e0e0e0;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{blog_content['title']}</h1>
        <div class="subtitle">{blog_content['subtitle']}</div>
        <div class="tags">
            {' '.join([f'<span class="tag">{tag}</span>' for tag in blog_content['metadata']['meta_tags']])}
        </div>
    </div>
"""

    # Add content sections
    for section in blog_content["content"]:
        html += f"""
    <div class="section">
        <h2>{section['heading']}</h2>
        <p>{section['content']}</p>
"""
        # Add subsections if they exist
        if "subsections" in section:
            for subsection in section["subsections"]:
                html += f"""
        <div class="subsection">
            <h3>{subsection['heading']}</h3>
            <p>{subsection['content']}</p>
        </div>
"""
        html += "    </div>\n"

    # Close HTML document
    html += """</body>
</html>"""

    return html
