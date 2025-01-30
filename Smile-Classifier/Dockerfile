# Use the official Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages and add the deadsnakes PPA
RUN apt-get update && \
    apt-get install -y software-properties-common curl libgl1-mesa-glx libglib2.0-0 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.10 and pip
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Install MySQL client
RUN apt-get install -y default-mysql-client

# Set the working directory
WORKDIR /FastAPI

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with error handling
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt || true && \
    python3.10 -m pip install --no-cache-dir uvicorn fastapi

# Copy the rest of the project files
COPY . .

# Download the wait-for-it script
ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /usr/local/bin/wait-for-it
RUN chmod +x /usr/local/bin/wait-for-it

# Set environment variables
ENV MYSQL_HOST=mysql_container
ENV MYSQL_PORT=3306
ENV MYSQL_USER=root
ENV MYSQL_PASSWORD=163801
ENV MYSQL_DB=smiley

# Expose the port
EXPOSE 80

# Run the application
CMD ["wait-for-it", "mysql_container:3306", "--timeout=30", "--", "python3.10", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]