from sqlalchemy import create_engine


DATABASE_URL = (
    "mysql+pymysql://"
    "admin:JustInTime321@"
    "lipspeak-db.c34k622s8fn6.ap-south-1.rds.amazonaws.com:3306/"
    "lipspeak"
)

engine = create_engine(
    DATABASE_URL,
)

with engine.connect() as connection:
    print(
        "Database Connected Successfully!"
    )