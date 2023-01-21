import MySQLdb
from sshtunnel import SSHTunnelForwarder
from uuid import getnode

with SSHTunnelForwarder(
         ('139.59.86.80', 22),
         ssh_password="giri@123",
         ssh_username="root",
         remote_bind_address=('127.0.0.1', 3306)) as server:
    conn = MySQLdb.connect(host='127.0.0.1',
                           port=server.local_bind_port,
                           user='root',
                           passwd='ASSpeed123@',
                           db='ATCC')
    cursor = conn.cursor()
    cursor.execute("select * from Cameras;")
    print cursor.fetchall()
    cursor.execute("select * from Machines;")
    for row in cursor.fetchall():
        print getnode() == row[-1]
    
    cursor.close()
    conn.close()
