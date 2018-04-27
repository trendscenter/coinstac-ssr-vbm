FROM coinstac/coinstac-base-python-stream

# Set the working directory
WORKDIR /computation

# Copy the current directory contents into the container
COPY requirements.txt /computation

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install scikit-learn
RUN pip install nose
RUN pip install matplotlib

# Copy the current directory contents into the container
COPY . /computation
