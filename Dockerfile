# Use an official tensorflow with gpu runtime as parent image
FROM gcr.io/tensorflow/tensorflow:1.4.0-gpu

# Set the working directory to /dense
WORKDIR /dense

# Copy the current directory contents into the container at /dense
ADD . /dense

# Install any needed packages specified in requirements.txt
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get update -y \
  && apt-get install tk-dev -y \
  && apt-get install python3-tk -y

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run densenet121.py when the container launches
CMD ["python", "densenet121.py"]
