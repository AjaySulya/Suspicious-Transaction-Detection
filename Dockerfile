#i use slim base for simpler image
FROM python:3.11-slim

# setting the working directory
WORKDIR /app
# installing the system dependencies if there is needed for the ( catBoost , LightGBM, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# here i am installing the production dependenciess only 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# here i am copy the project file 
COPY . .

# expose port (flask,restapi by defaul)
EXPOSE 5000

# Starting the production server
CMD ["gunicorn","app:app","--bind","0.0.0.0:5000"]

# Building docker container and run it on port 5000
# docker build -t paysim-app -f Dockerfile .
# docker run -p 5000:5000 paysim-app
