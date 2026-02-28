import sqlite3

def check_session_id(session_id):
    
    #连接数据库
    con=sqlite3.connect('chat_history.db')
    cursor = con.cursor()
    
    valid_table_exists_sql = "select count(*) from sqlite_master where type='table' and name='message_store';"
    res = cursor.execute(valid_table_exists_sql)
    
    if res.fetchone()[0] == 0:
        return False
        
    search_session_id_sql = f"select distinct session_id from message_store "
    res=cursor.execute(search_session_id_sql)
    
    all_session_id=res.fetchall()
    
    #关闭数据库连接

    cursor.close()
    con.close()
    
    return [item[0] for item in all_session_id ]

if __name__ == "__main__":
    print(check_session_id('abc123'))  # 测试函数，检查会话 ID 是否存在
    print(check_session_id('abc456'))  # 测试函数，检查不存在的会话 ID