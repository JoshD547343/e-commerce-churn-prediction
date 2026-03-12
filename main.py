from app import app, load_model
from db import init_db

if __name__ == "__main__":
    init_db()
    load_model()

    print("Starting server at http://localhost:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )