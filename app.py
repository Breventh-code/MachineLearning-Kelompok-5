"""Menjalankan aplikasi SmartBearing CWRU."""
from backend import create_app

app = create_app()

if __name__ == "__main__":
    # Debug hanya untuk pengembangan lokal, jangan digunakan di production.
    app.run(host="127.0.0.1", port=5000, debug=True)
