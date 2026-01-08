import os

STRUCTURE = {
    "app.py": "",
    "requirements.txt": "",
    "templates": {
        "index.html": "<!-- main UI -->\n",
    },
    "static": {
        "style.css": "/* styles */\n",
        "app.js": "// frontend logic\n",
        "sounds": {},  # optional folder
    },
    "core": {
        "__init__.py": "",
        "detector.py": "# gesture detection logic\n",
        "gestures.py": "# gesture mapping logic\n",
        "utils.py": "# helper utilities\n",
    },
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            if not os.path.exists(path):   # already থাকলে touch করবে না
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

if __name__ == "__main__":
    create_structure(".", STRUCTURE)
    print("✅ speak-without-words structure ready (existing files untouched)")
