from sqlalchemy import create_engine, text


SERVER = "DESKTOP-C0V38AA"
DATABASE = "REAL_ESTATE_MARKET_INTELLIGENCE"
DRIVER = "ODBC Driver 17 for SQL Server"


CONNECTION_STRING = (
    f"mssql+pyodbc://@{SERVER}/{DATABASE}"
    f"?driver={DRIVER.replace(' ', '+')}"
    "&trusted_connection=yes"
    "&TrustServerCertificate=yes"
)


engine = create_engine(
    CONNECTION_STRING,
    pool_pre_ping=True,
)


def get_connection():
    return engine.connect()


def test_connection():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT DB_NAME() AS current_database;")
        )

        return result.scalar()