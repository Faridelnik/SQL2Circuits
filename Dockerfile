# build with: docker build --rm -t sql2circuits_image -f SQL2Circuits/Dockerfile .
# run with: docker run -it -v "$(pwd)/SQL2Circuits:/qc4db/SQL2Circuits" --gpus 1 sql2circuits_image bash

# Load base image:
FROM ubuntu:22.04
ENV lsb_release_cs=jammy

#FROM nvidia/cuda:12.3.1-base-ubuntu22.04
#FROM nvidia/cuda:11.8.0-base-ubuntu22.04
#ENV lsb_release_cs=jammy

USER root
SHELL ["/bin/bash", "-c"]

# default number of cores to run the build/install with
ARG N_PROC=0
RUN echo -e '#!/bin/bash\nif [[ ${N_PROC} -eq 0 ]]\nthen\n\tgetconf _NPROCESSORS_ONLN\nelse\n\techo ${N_PROC}\nfi' > /usr/bin/getNumCores \ 
    && chmod +x /usr/bin/getNumCores \
    && echo "Compiling with $(getNumCores)/$(getconf _NPROCESSORS_ONLN) threads" 

# Install extra packages
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections \
    && apt update \
    && apt -y install build-essential git wget ca-certificates python3-pip python3.10-dev dnsutils iputils-ping net-tools cmake g++-12 vim\
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && which g++ \
    && g++ --version

# Install postgresql
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
    && bash -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ $lsb_release_cs-pgdg main" >> /etc/apt/sources.list.d/pgdg.list' \
    && apt update \
    && apt -y install postgresql-14 postgresql-contrib-14 libpq-dev postgresql-server-dev-14 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install psycopg2 -I --no-cache-dir

# - - - - - - - - - - - - - - - - - - - - postgreSQL setup - - - - - - - - - - - - - - - - - - # 
RUN    echo "host all  all    0.0.0.0/0  md5"  >> /etc/postgresql/14/main/pg_hba.conf \
    && echo "listen_addresses='*'"             >> /etc/postgresql/14/main/postgresql.conf

# add postgreSQL c++ interface library
RUN wget "https://github.com/jtv/libpqxx/archive/refs/tags/7.7.5.tar.gz" -O libpqxx-7.7.5.tar.gz \
    && tar xvfz libpqxx-7.7.5.tar.gz \
    && cd libpqxx-7.7.5 \
    && cmake -S. -Bbuild \
    && cmake --build build -- -j$(getNumCores) \
    && cmake --install build \
    && rm -rf ../libpqxx-7.7.5*

# copy database
WORKDIR /qc4db
COPY dataBase ./dataBase

WORKDIR /qc4db
# init users and data bases for tests
USER postgres
RUN /etc/init.d/postgresql start \
    && psql --command "CREATE USER docker WITH SUPERUSER PASSWORD 'docker';" \ 
    && psql --command "ALTER USER postgres PASSWORD 'test_123';" \
    && createdb db_test \
    && psql db_test < dataBase/db_test.pgsql \
    && createdb F1Data \
    && psql -d F1Data -f /qc4db/dataBase/ergastF1/create.sql \
    && psql -d F1Data -f /qc4db/dataBase/ergastF1/load.sql

RUN pip install --upgrade pip

USER root

# Create IMDB database -----------------------------------------------------------------------

WORKDIR /qc4db/IMDBdataset
RUN wget -O /qc4db/IMDBdataset/imdb_pg11 https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/2QYZBT/TGYUNU
USER postgres
RUN /etc/init.d/postgresql start \
    && createdb imdb \
    && pg_restore -x --no-owner -d imdb -1 /qc4db/IMDBdataset/imdb_pg11


# Install packages ---------------------------------------------------------------------------
USER root
RUN pip install antlr4-tools==0.2 antlr4-python3-runtime==4.11.1 scikit-learn==1.3.2 discopy==1.1.4 \
    optax==0.1.9 lambeq==0.3.3 matplotlib==3.7.3 noisyopt==0.2.2 numpy==1.26.4 PennyLane==0.34.0 \
    psycopg2_binary==2.9.9 sympy==1.12 seaborn==0.13.2 chex==0.1.85

# torch >=1.12.1

RUN pip install jax==0.4.24 jaxlib==0.4.24+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install -U nvidia-cudnn-cu12==8.9.2.26

#RUN pip install -U "jax[cuda12]"

#RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases // jax-0.4.30 jaxlib-0.4.30 nvidia-cudnn-cu12-9.2.1.18

#pip3 install -U nvidia-cudnn-cu12==8.9.7.29
#RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
 
# RUN pip install antlr4-tools==0.2 antlr4-python3-runtime==4.11.1 scikit-learn==1.3.2 discopy==1.1.4 \
#     optax==0.1.7 lambeq==0.3.3 matplotlib==3.7.3 noisyopt==0.2.2 numpy==1.23.5 PennyLane==0.32.0 \
#     psycopg2_binary==2.9.9 sympy==1.12 seaborn==0.13.2

# RUN pip install -U scikit-learn
# RUN pip install jax==0.4.29 jaxlib==0.4.29+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip install jax==0.4.7 jaxlib==0.4.7+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html chex==0.1.7

# start the training ---------------------------------------------------------------------------
CMD ["cd ..", "/etc/init.d/postgresql start", "cd SQL2Circuits/sql2circuits", "python3 main.py"]













