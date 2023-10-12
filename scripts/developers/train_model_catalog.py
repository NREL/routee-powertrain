import pandas as pd
import glob

from pathlib import Path

import random

from tqdm import tqdm

import nrel.routee.powertrain as pt

from nrel.routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

MODEL_BASE_PATH = "/projects/mbap/data/fastsim-results/wm1-fastsim-results/"
ALL_FILES = glob.glob(f"{MODEL_BASE_PATH}**/**/*.parquet")

OUTPUT_BASE_PATH = Path("/projects/mbap/data/routee-results/model-catalog/")
if not OUTPUT_BASE_PATH.exists():
    OUTPUT_BASE_PATH.mkdir()


LOW_MILES_THRESHOLD = 50 / 5280  # 50 feet
HIGH_MILES_THRESHOLD = 0.5
HIGH_GGE_PER_MILE_THRESHOLD = 0.5  # 2 mpg
HIGH_KWH_PER_MILE_THRESHOLD = 5
LOW_KWH_PER_MILE_THRESHOLD = -5
KWH_TO_GGE = 0.029931063
HIGH_GRADE_THRESHOLD = 0.4
LOW_GRADE_THRESHOLD = -0.4

FILE_LIMIT = 10
N_CORES = 80

models = [
    "2012_Ford_Focus",
    "2012_Ford_Fusion",
    "2016_AUDI_A3_4cyl_2WD",
    "2016_BMW_328d_4cyl_2WD",
    # "2016_BMW_i3_REx_PHEV_Charge_Depleting",
    # "2016_BMW_i3_REx_PHEV_Charge_Sustaining",
    "2016_CHEVROLET_Malibu_4cyl_2WD",
    "2016_CHEVROLET_Spark_EV",
    # "2016_CHEVROLET_Volt_Charge_Depleting",
    # "2016_CHEVROLET_Volt_Charge_Sustaining",
    "2016_FORD_C-MAX_HEV",
    # "2016_FORD_C-MAX_(PHEV)_Charge_Depleting",
    # "2016_FORD_C-MAX_(PHEV)_Charge_Sustaining",
    "2016_FORD_Escape_4cyl_2WD",
    "2016_FORD_Explorer_4cyl_2WD",
    "2016_HYUNDAI_Elantra_4cyl_2WD",
    # "2016_HYUNDAI_Sonata_PHEV_Charge_Depleting",
    # "2016_HYUNDAI_Sonata_PHEV_Charge_Sustaining",
    "2016_Hyundai_Tucson_Fuel_Cell",
    "2016_KIA_Optima_Hybrid",
    "2016_Leaf_24_kWh",
    "2016_MITSUBISHI_i-MiEV",
    "2016_Nissan_Leaf_30_kWh",
    "2016_TESLA_Model_S60_2WD",
    "2016_TOYOTA_Camry_4cyl_2WD",
    "2016_TOYOTA_Corolla_4cyl_2WD",
    "2016_TOYOTA_Highlander_Hybrid",
    "2016_Toyota_Prius_Two_FWD",
    "2017_CHEVROLET_Bolt",
    "2017_Maruti_Dzire_VDI",
    # "2017_Prius_Prime_Charge_Depleting",
    # "2017_Prius_Prime_Charge_Sustaining",
    "2017_Toyota_Highlander_3.5_L",
    "2020_Chevrolet_Colorado_2WD_Diesel",
    "2020_VW_Golf_1.5TSI",
    "2020_VW_Golf_2.0TDI",
    "2021_Fiat_Panda_Mild_Hybrid",
    "2021_Peugot_3008",
    "2022_Ford_F-150_Lightning_4WD",
    "2022_Renault_Zoe_ZE50_R135",
    "2022_Tesla_Model_3_RWD",
    "2022_Tesla_Model_Y_RWD",
    "2022_Toyota_Avanza_E",
    "2022_Toyota_Yaris_Hybrid_Mid",
    "2022_Volvo_XC40_Recharge_twin",
    "2023_Mitsubishi_Pajero_Sport",
    "Maruti_Swift_4cyl_2WD",
    "Nissan_Navara",
    "Renault_Clio_IV_diesel",
    "Renault_Megane_1.5_dCi_Authentique",
    "Toyota_Hilux_Double_Cab_4WD",
    "Toyota_Mirai",
]


def load_df(file, trip_start_id):
    df = pd.read_parquet(
        file,
        columns=[
            "journey_id",
            "speed_mph",
            "grade_dec",
            "fs_kwh_out_ach",
            "ess_kwh_out_ach",
            "miles",
        ],
    )

    df = df[(df.miles > LOW_MILES_THRESHOLD) & (df.miles < HIGH_MILES_THRESHOLD)]
    df = df[
        (df.grade_dec > LOW_GRADE_THRESHOLD) & (df.grade_dec < HIGH_GRADE_THRESHOLD)
    ]
    df["gge"] = df.fs_kwh_out_ach * KWH_TO_GGE
    df["kwh"] = df.ess_kwh_out_ach

    df["gal_per_miles"] = df.gge / df.miles
    df = df[df.gal_per_miles < HIGH_GGE_PER_MILE_THRESHOLD]

    df["kwh_per_mile"] = df.ess_kwh_out_ach / df.miles
    df = df[
        (df.kwh_per_mile > LOW_KWH_PER_MILE_THRESHOLD)
        & (df.kwh_per_mile < HIGH_KWH_PER_MILE_THRESHOLD)
    ]

    df["trip_id"] = pd.factorize(df.journey_id)[0]
    df["trip_id"] = df.trip_id + (trip_start_id + 1)

    df = df.drop(
        columns=[
            "gal_per_miles",
            "fs_kwh_out_ach",
            "ess_kwh_out_ach",
            "journey_id",
            "kwh_per_mile",
        ]
    )

    df = df.astype(
        {
            "speed_mph": "float32",
            "grade_dec": "float32",
            "miles": "float32",
            "gge": "float32",
        }
    )

    max_trip_id = df.trip_id.iloc[-1]

    return df.reset_index(drop=True), max_trip_id


def load_all_files(files, file_limit=FILE_LIMIT):
    files_to_load = random.choices(files, k=file_limit)
    dfs = []
    trip_start_id = 0
    for file in files_to_load:
        df, trip_start_id = load_df(file, trip_start_id)
        dfs.append(df)
    df = pd.concat(dfs, copy=False)
    return df


feature_set_1 = [pt.DataColumn(name="speed_mph", units="mph")]
feature_set_2 = [
    pt.DataColumn(name="speed_mph", units="mph"),
    pt.DataColumn(name="grade_dec", units="decimal"),
]
features = [
    feature_set_1,
    feature_set_2,
]

distance = pt.DataColumn(name="miles", units="miles")


def train_model(model_name):
    model_files = list(filter(lambda f: model_name in f, ALL_FILES))

    df = load_all_files(model_files)

    print(f"loaded {len(df)} samples")

    if df.gge.sum() > 0 and df.kwh.sum() < 0.001:
        powertrain_type = pt.PowertrainType.ICE
        energy_target = pt.DataColumn(
            name="gge",
            units="gallons_gasoline",
        )
    elif df.gge.sum() < 0.001 and df.kwh.sum() > 0:
        powertrain_type = pt.PowertrainType.BEV
        energy_target = pt.DataColumn(
            name="kwh",
            units="kwh",
        )
    else:
        raise ValueError(f"Dual fuel powertrains not yet supported for {model_name}")

    config = pt.ModelConfig(
        vehicle_description=model_name,
        powertrain_type=powertrain_type,
        feature_sets=features,
        distance=distance,
        target=energy_target,
        test_size=0.2,
    )

    trainer = SklearnRandomForestTrainer(cores=80)

    vehicle = trainer.train(df, config)

    return vehicle


if __name__ == "__main__":
    for m_name in tqdm(models):
        try:
            vehicle = train_model(m_name)
        except ValueError as e:
            print(f"error for {m_name}, skipping")
            print(e)
            continue

        outfile = OUTPUT_BASE_PATH / f"{m_name}.json"

        vehicle.to_file(outfile)
