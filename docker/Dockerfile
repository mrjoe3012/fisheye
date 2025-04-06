FROM ubuntu:latest
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt update
RUN apt install -y python3 python3-venv
RUN python3 -m venv /venv
COPY . /fisheye
RUN /bin/sh -c ". /venv/bin/activate && \
    pip install --no-cache-dir -e /fisheye/fisheye_python && \
    pip install --no-cache-dir -r /fisheye/fisheye_python/dev_requirements.txt"
RUN rm -rf /fisheye /var/lib/apt/lists/*
CMD ["/bin/sh", "-c", ". /venv/bin/activate && exec /bin/sh"]
