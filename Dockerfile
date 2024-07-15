# Pull the latest python image.

FROM python

# Set the maintainer.

LABEL maintainer="eduard_nicolae@yahoo.com"

# Update packages.

RUN apt-get update

# Install Node.js.

RUN apt-get install nodejs -y

# Install NPM for package installer.

RUN apt-get install npm -y

# Install PM2 for process manager.

RUN npm install pm2 -g

# Upgrade PIP installer.

RUN pip install --upgrade pip

# Set work directory.

WORKDIR /TranslationService

# Copies all files from local storage to the selected working directory.

COPY . .

# Install app requirements.

RUN pip install -r requirements.txt