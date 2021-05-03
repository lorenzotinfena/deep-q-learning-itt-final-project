FROM python

RUN apt upgrade
RUN apt update
RUN pip install --upgrade pip

RUN apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python-opengl \
       texlive-xetex \
       texlive-fonts-recommended \
       texlive-generic-recommended

ADD ./.devcontainer/pip-requirements.txt /tmp/
RUN pip install -r /tmp/pip-requirements.txt
RUN rm /tmp/pip-requirements.txt

# fix
RUN cd /usr/local/lib/ \
       git clone https://github.com/openai/gym.git \
       pip install ./gym