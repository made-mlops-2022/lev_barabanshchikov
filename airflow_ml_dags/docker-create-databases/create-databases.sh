#!/bin/bash

set -e
set -u

function create_db_and_user() {
  local name=$1
  echo "Creating database and user: '$name'"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE USER $name;
    CREATE DATABASE $name;
    GRANT ALL PRIVILEGES ON DATABASE $name TO $name;
EOSQL
}

if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
  echo "Creating multiple databases: $POSTGRES_MULTIPLE_DATABASES"
  for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
    create_db_and_user $db
  done
  echo "Multiple databases created"
fi
