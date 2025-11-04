# Using an offical lightweight python image 
FROM python:3.12-slim

#setting working directory
WORKDIR /app

#Copying the files
COPY . /app

#Install system dependencies 
RUN apt-get update && apt-get install -y \ 
    build-essential \ 
    && rm -rf /var/lib/apt/lists/*

#Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Set streamlit default port
EXPOSE 8501

#  Set Streamlit environment variables (optional)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_HEADLESS=true

#Run Your Streamlit app
CMD ["streamlit", "run", "app/dashboard.py"]
