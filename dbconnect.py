import mysql.connector
import pandas as pd
from sqlalchemy import create_engine


def load_from_db(sql):
  mydb = mysql.connector.connect(
    host="db4free.net",
    user="app_root",
    passwd="mysql@12345678",
    database="hoian_travel"
  )

  mycursor = mydb.cursor()
  mycursor.execute(sql)
  myresult = mycursor.fetchall()
  mycursor.close()

  return myresult


def read_data_from_db(sql):
    db_connection_str = 'mysql+pymysql://app_root:mysql@12345678@db4free.net/hoian_travel'
    db_connection = create_engine(db_connection_str, pool_recycle=3600)

    df = pd.read_sql(sql, db_connection)
    pd.read_sql(sql, db_connection)
    pd.set_option('display.expand_frame_repr', False)

    return df