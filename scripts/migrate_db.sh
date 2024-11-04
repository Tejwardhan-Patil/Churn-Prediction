#!/bin/bash

# Script to manage database migrations

# Variables
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="churn_prediction"
DB_USER="db_user"
DB_PASSWORD="db_password"
MIGRATIONS_DIR="./migrations"
SQL_FILE="$MIGRATIONS_DIR/migration.sql"

# Function to check if PostgreSQL is running
function check_postgres {
    pg_isready -h $DB_HOST -p $DB_PORT
    if [ $? -ne 0 ]; then
        echo "PostgreSQL is not running on $DB_HOST:$DB_PORT"
        exit 1
    fi
}

# Function to apply migration
function apply_migration {
    echo "Applying migration from $SQL_FILE to $DB_NAME"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f $SQL_FILE
    if [ $? -ne 0 ]; then
        echo "Migration failed"
        exit 1
    fi
    echo "Migration applied successfully"
}

# Function to rollback the last migration
function rollback_migration {
    echo "Rolling back last migration"
    ROLLBACK_FILE="$MIGRATIONS_DIR/rollback.sql"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f $ROLLBACK_FILE
    if [ $? -ne 0 ]; then
        echo "Rollback failed"
        exit 1
    fi
    echo "Rollback applied successfully"
}

# Parse command line arguments
case $1 in
    migrate)
        check_postgres
        apply_migration
        ;;
    rollback)
        check_postgres
        rollback_migration
        ;;
    *)
        echo "Usage: $0 {migrate|rollback}"
        exit 1
        ;;
esac