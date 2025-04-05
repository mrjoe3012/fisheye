FROM python:3.10.14-alpine
COPY . /fisheye
RUN python -m venv /venv
RUN /bin/sh -c ". /venv/bin/activate && \
    pip install -e /fisheye/fisheye_python"
RUN rm -rf /fisheye
CMD ["/bin/sh", "-c", ". /venv/bin/activate && exec /bin/sh"]
