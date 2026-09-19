from database import test_connection


if __name__ == "__main__":
    database_name = test_connection()

    print(f"Connected successfully to: {database_name}")