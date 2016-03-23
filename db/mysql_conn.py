import mysql.connector

class MysqlDb(object):
    table_name = 'training_data_e1'
    
    def __init__(self):
        self.conn = mysql.connector.connect(user='root', password='Rvpooh123',
                              host='127.0.0.1',
                              database='sdc')
        
    def get_75_row(self, cred_value):
        cur = self.conn.cursor()
        cur.execute("select id from "+self.table_name+" where cred_value = '"+cred_value+"' order by rand() limit 75")
        ret_data = []
        for row in cur.fetchall():
            ret_data.append(row[0])
        return ret_data

    def get_25_row(self, cred_value):
        cur = self.conn.cursor()
        cur.execute("select id from "+self.table_name+" where cred_value = '"+cred_value+"' order by rand() limit 25")
        ret_data = []
        for row in cur.fetchall():
            ret_data.append(row[0])
        return ret_data
    
    def get_n_row(self, cred_value, n):
        cur = self.conn.cursor()
        cur.execute("select id from "+self.table_name+" where cred_value = '"+cred_value+"' order by rand() limit "+str(n))
        ret_data = []
        for row in cur.fetchall():
            ret_data.append(row[0])
        return ret_data
            
    def get_data(self):
        cur = self.conn.cursor()
        cur.execute("select id, message from "+self.table_name)
        ret_data = []
        for row in cur.fetchall():
            ret_data.append(row)
        cur.close()
        self.conn.close()
        return ret_data
    
    def get_feature_by_row(self, lst_row_id):
        cur = self.conn.cursor()
#         sql = '''
#         select likes, shares, comments, hashtags, images, vdo, url, word_in_dict, word_outside_dict, num_of_number_in_sentense
#         from {}
#         where id in {}
#         '''.format(self.table_name, lst_row_id)
        sql = '''
        select likes, shares, comments, hashtags, images, vdo, url, word_in_dict, word_outside_dict, num_of_number_in_sentense, app_sender,
        share_with_location, share_with_non_location, tag_with, feeling_status, share_public, share_only_friend, word_count, character_length,
        question_mark, exclamation_mark
        from {}
        where id in {}
        '''.format(self.table_name, lst_row_id)
        cur.execute(sql)
        ret_data = []
        for row in cur.fetchall():
            row_data = list(row)
            ret_data.append(row_data)
        return ret_data
    
    def get_label_data(self, lst_id):        
        cur = self.conn.cursor()
        sql = '''
        select cred_value
        from {}
        where id in {}
        '''.format(self.table_name, lst_id)
        cur.execute(sql)
        ret_data = []
        for row in cur.fetchall():
            if row[0] == 'yes':
                ret_data.append(1)
            else:
                ret_data.append(0)
        return ret_data
    
    def row_count(self, rec_id):
        cur = self.conn.cursor()
        sql = 'select count(*) from training_data_e1 where id = '+rec_id
        cur.execute(sql)
        row = cur.fetchone()
        return row[0]
    
if __name__ == '__main__':
    obj = MysqlDb()
    import filter01
    measure_data = filter01.read_line('../data/measurement.txt')
    un_measure = filter01.read_line('../data/unmeasurement.txt')
    for data in un_measure:
        row_count = obj.row_count(data)
        if row_count == 0:
            print 'id ', data, ' count ',row_count
    


