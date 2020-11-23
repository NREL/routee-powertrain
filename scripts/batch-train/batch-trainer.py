import io
import logging
import multiprocessing as mp
import os
import tempfile

import pandas as pd
import sqlalchemy as sql
import yaml

from powertrain.core.features import Feature, FeaturePack
from powertrain.core.model import Model
from powertrain.estimators.explicit_bin import ExplicitBin
from powertrain.estimators.linear_regression import LinearRegression
from powertrain.estimators.random_forest import RandomForest

logging.basicConfig(filename='batch_run.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s')

logging.info('RouteE batch run START')


def load_config(config_file):
    """
    Load the user config file, config.yml
    This is where all configurations for the batch run are stored.
    """

    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def read_sql_inmem_uncompressed(query, db_engine):
    copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
        query=query, head="HEADER"
    )
    conn = db_engine.raw_connection()
    cur = conn.cursor()
    store = io.StringIO()
    cur.copy_expert(copy_sql, store)
    store.seek(0)
    df = pd.read_csv(store)
    return df


def read_sql_tmpfile(query, db_engine):
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
            query=query, head="HEADER"
        )
        conn = db_engine.raw_connection()
        cur = conn.cursor()
        cur.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
        return df


def train_routee_model(tuple_in):
    # Read vehicle specs
    config = tuple_in[0]
    db_name = tuple_in[1]

    # Read FASTSim "link" results

    # Train model(s)

    # Write output as JSON and CSV where applicable

    logging.info('    Training on data from {}'.format(db_name))

    vehicle_name = db_name.replace('.db', '')

    features = (
        Feature('gpsspeed', units='mph'),
        Feature('grade', units='percent_0_100'),
    )
    distance = Feature('miles', units='mi')

    sql_con = sql.create_engine('sqlite:///' + config['fastsim_results_path'] + db_name)

    logging.info('    Reading SQLite')

    quer = """
    SELECT sampno, vehno, tripno, gpsspeed, miles, grade, gge, esskwhoutach
    FROM links
    LIMIT 100000
    """

    df = pd.read_sql(quer, sql_con)
    df['grade'] = df.grade.apply(lambda x: x * 100)

    # Determine fuel type - gge must come first to catch HEVs
    if df.gge.sum() > 0:
        energy = Feature('gge', units='gallons')
    elif df.esskwhoutach.sum() > 0:
        energy = Feature('esskwhoutach', units='kwh')
    else:
        raise RuntimeError('There is no energy in this data file..')

    train_df = df[['miles', 'gpsspeed', 'grade', energy.name]].dropna()
    feature_pack = FeaturePack(features, distance, energy)

    logging.info('    Training LinReg')
    ln_e = LinearRegression(feature_pack=feature_pack)
    logging.info('    Training RF')
    rf_e = RandomForest(feature_pack=feature_pack)
    logging.info('    Training Exbin')
    eb_e = ExplicitBin(feature_pack=feature_pack)

    logging.info('    Dumping models')
    for e in (ln_e, rf_e, eb_e):
        m = Model(e, description=vehicle_name)
        m.train(train_df)
        m.to_json(config['routee_results_path'] + f"{vehicle_name}_{e.__class__.__name__}.json")
        m.to_pickle(config['routee_results_path'] + f"{vehicle_name}_{e.__class__.__name__}.pickle")

    return


if __name__ == '__main__':

    config = load_config('config.yml')

    # Initialize results location
    logging.info('Inititalizing results directory: %s' % config['routee_results_path'])

    if not os.path.exists(config['routee_results_path']):
        os.makedirs(config['routee_results_path'])

    # List FASTSim results DBs
    fs_results_dbs = os.listdir(config['fastsim_results_path'])
    fs_results_dbs = [fn for fn in fs_results_dbs if fn.endswith('.db')]

    tup_input = zip([config] * len(fs_results_dbs), fs_results_dbs)

    pool = mp.Pool(processes=config['n_cores'])
    pool.map(train_routee_model, tup_input)

    pool.close()
    pool.terminate()
    pool.join()
