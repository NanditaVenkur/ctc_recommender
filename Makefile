ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
    RM := del /Q
    RMDIR := rmdir /S /Q
else
    PYTHON := .venv/bin/python
    RM := rm -f
    RMDIR := rm -rf
endif

APP := backend.main:app
HOST := 127.0.0.1
PORT := 8000

.PHONY: help install run dev recreate-db clean test

help:
	@echo ""
	@echo "CTC Offer Intelligence"
	@echo ""
	@echo "Available targets:"
	@echo "  make install       Install dependencies"
	@echo "  make run           Start application"
	@echo "  make dev           Start with auto reload"
	@echo "  make recreate-db   Delete SQLite database"
	@echo "  make clean         Remove cache files"
	@echo "  make test          Run tests"

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m uvicorn $(APP) --host $(HOST) --port $(PORT)

dev:
	$(PYTHON) -m uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

recreate-db:
ifeq ($(OS),Windows_NT)
	@if exist ctc_recommender.sqlite3 $(RM) ctc_recommender.sqlite3
else
	$(RM) ctc_recommender.sqlite3
endif
	@echo "Database removed."
	@echo "It will be recreated on next application startup."

clean:
ifeq ($(OS),Windows_NT)
	@if exist __pycache__ $(RMDIR) __pycache__
	@for /d /r %i in (__pycache__) do @if exist "%i" $(RMDIR) "%i"
	@del /S /Q *.pyc 2>nul || exit 0
else
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
endif

test:
	$(PYTHON) -m pytest
