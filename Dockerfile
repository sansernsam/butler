FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000

EXPOSE ${PORT}

# Update the command to explicitly use start
ENTRYPOINT ["python", "butler.py", "start"]
