# run from top level folder qc4db
# build takes ~200s (on MBAm2 amd64 emulated) and ~70s (on MBAm2 arm64 native)
# substitute PLATFORM with linux/arm64 or linux/amd64 and DOCKERFILE 
# build with: docker buildx build --platform PLATFORM --progress=plain -t server -f docker/DOCKERFILE .
# run with: docker run  --platform PLATFORM -P --name pgs -it --rm server

# Load base image:
FROM ubuntu:22.04
ENV lsb_release_cs=jammy

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
    && apt -y install build-essential git wget ca-certificates python3-pip dnsutils iputils-ping net-tools cmake g++-12 vim\
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

# create and populate IMDB database
# USER root
# WORKDIR /qc4db/frozendata
# RUN wget ftp://ftp.fu-berlin.de/misc/movies/database/frozendata/*gz

# USER postgres
# RUN /etc/init.d/postgresql start \
#     && createdb imdbload

# WORKDIR /qc4db
# #RUN pip install cinemagoer
# COPY dataBase/cinemagoer ./cinemagoer
# RUN pip install SQLAlchemy==2.0.25
# WORKDIR /qc4db/cinemagoer/bin
# RUN /etc/init.d/postgresql start \
#     && python3 imdbpy2sql.py -d /qc4db/frozendata -u postgresql://postgres:test_123@localhost:5432/imdbload || true

# clone SQL2Circuits
USER root
WORKDIR /qc4db

RUN pip install antlr4-tools==0.2 antlr4-python3-runtime==4.11.1 scikit-learn==1.3.2 discopy==1.1.4 \
    jax==0.4.23 jaxlib==0.4.23 optax==0.1.7 lambeq==0.3.3 matplotlib==3.7.3 noisyopt==0.2.2 numpy==1.23.5 PennyLane==0.32.0 \
    psycopg2_binary==2.9.9 sympy==1.12 seaborn==0.13.2
RUN pip install -U scikit-learn


