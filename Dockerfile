# Use an official Python runtime as a parent image
FROM python:3.5-slim

# Set the working directory to /dense
WORKDIR /dense

# Copy the current directory contents into the container at /dense
ADD . /dense

# Install any needed packages specified in requirements.txt
RUN apt-get update -y && apt-get install tk-dev -y
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run densenet121.py when the container launches
CMD ["python", "densenet121.py"]
