"""PostgreSQL connection for the ui-chart service.

Read-only visualisation tool — a fresh connection per request is fine, no pool.
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv('config/confluent.env')


def get_conn():
    return psycopg2.connect(
        host=os.getenv('PG_HOST', 'localhost'),
        port=os.getenv('PG_PORT', 5432),
        dbname=os.getenv('PG_DB', 'spendlabel'),
        user=os.getenv('PG_USER', 'postgres'),
        password=os.getenv('PG_PASSWORD', ''),
    )
