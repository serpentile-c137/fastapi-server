# Start from the Python 3.12.4 base image
FROM python:3.12.4

# Install system dependencies (including libGL and build essentials)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    build-essential \
    && rm -rf /var/lib/apt/lists/*  # Clean up the apt cache to reduce image size

# Create a new non-root user and switch to that user
RUN useradd -m -u 1000 user
USER user

# Set the PATH to include local user binaries
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file with proper ownership
COPY --chown=user ./requirements.txt requirements.txt

# Install Python dependencies from the requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application files with proper ownership
COPY --chown=user . /app

# Command to run the application using Gunicorn
# CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
