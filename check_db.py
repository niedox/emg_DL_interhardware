from bioml import Data_Selector as DS

params = {
    'dbname': 'leonardo',
    'user': 'postgres',
    'host': '128.178.51.53',
    'port': 5432
    }

DB = DS.get_db_names(params)
print('Available Databases:', [db['label'] for db in DB])


if params['dbname'] != '':
    recs_dict = DS.get_recordings(params)
    recs = [rec['label'] for rec in recs_dict]
    print('Available recordings:', recs)