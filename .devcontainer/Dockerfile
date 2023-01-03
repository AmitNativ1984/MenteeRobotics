ARG BASE_IMAGE
FROM $BASE_IMAGE as dev

RUN pip install "ipywidgets>=7.0,<8.0" widgetsnbextension
RUN jupyter nbextension enable --py widgetsnbextension

