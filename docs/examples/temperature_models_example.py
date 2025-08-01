"""
# Temperature Models Example

This example demonstrates how to use models with temperature as a feature in the Routee Powertrain library.
"""

# %%
import nrel.routee.powertrain as pt
import pandas as pd
import matplotlib.pyplot as plt

# %%
"""
To demonstrate the use of temperature models, we will load a standard Tesla Model 3 RWD model (without temperature as a feature) and two temperature models (steady-state and transient) for the same vehicle. 
We will then predict energy consumption over a sample route at different ambient temperatures (72°F and 32°F) and compare the results.

It's important to understand the distinction between steady-state and transient temperature models:
- **Steady-State Temperature Models**: These models assume that the vehicle's thermal conditions have stabilized to a constant state. 
    They are typically used for longer trips where the vehicle has had sufficient time to reach thermal equilibrium with its environment. 
    In this example, we will use the steady-state model for the portion of the trip after the first 5 minutes at 32°F.
- **Transient Temperature Models**: These models account for the period when the vehicle is still adjusting to the ambient temperature.
    For example, when a vehicle starts a trip in cold weather and has been sitting outside, it takes some time for the battery and cabin to warm up.
"""
# %%
tesla = pt.load_model("2022_Tesla_Model_3_RWD")
tesla_with_temp_steady = pt.load_model(
    "../../../../Downloads/routee-temperature-models/2022_Tesla_Model_3_RWD_0F_110F_steady.json"
)
tesla_with_temp_transient = pt.load_model(
    "../../../../Downloads/routee-temperature-models/2022_Tesla_Model_3_RWD_0F_110F_transient.json"
)
# %%
"""
Load a sample route and prepare it for prediction.
"""
# %%
sample_route = pt.load_sample_route()
sample_route["time_minutes"] = (
    sample_route["distance"] / sample_route["speed_mph"]
) * 60
sample_route["cummulative_time_minutes"] = sample_route["time_minutes"].cumsum()
sample_route["cummulative_distance"] = sample_route["distance"].cumsum()
# %%
"""
Set the ambient temperature for the route.
"""
# %%
sample_route_72F = sample_route.copy()
sample_route_72F["ambient_temp_f"] = 72

sample_route_32F = sample_route.copy()
sample_route_32F["ambient_temp_f"] = 32

# %%
"""
For the 32°F route, we will use the transient model for the first 5 minutes and the steady-state model for the remainder of the trip.
"""
# %%
sample_route_32F_transient = sample_route_32F[
    sample_route_32F["cummulative_time_minutes"] <= 5
]
sample_route_32F_steady = sample_route_32F[
    sample_route_32F["cummulative_time_minutes"] > 5
]
# %%
"""
Predict energy consumption using the different models and ambient temperatures.
"""
# %%
energy = tesla.predict(sample_route, feature_columns=["speed_mph", "grade_percent"])
energy_with_temp_72F = tesla_with_temp_steady.predict(
    sample_route_72F, feature_columns=["speed_mph", "grade_percent", "ambient_temp_f"]
)
energy_with_temp_32F_transient = tesla_with_temp_transient.predict(
    sample_route_32F_transient,
    feature_columns=["speed_mph", "grade_percent", "ambient_temp_f"],
)
energy_with_temp_32F_steady = tesla_with_temp_steady.predict(
    sample_route_32F_steady,
    feature_columns=["speed_mph", "grade_percent", "ambient_temp_f"],
)
energy_with_temp_32F = pd.concat(
    [energy_with_temp_32F_transient, energy_with_temp_32F_steady]
)
# %%
"""
Now, we can compare the energy consumption results.
"""
# %%
energy["cummulative_energy_kwh"] = energy["kwh"].cumsum()
energy_with_temp_72F["cummulative_energy_kwh"] = energy_with_temp_72F["kwh"].cumsum()
energy_with_temp_32F["cummulative_energy_kwh"] = energy_with_temp_32F["kwh"].cumsum()
# %%
plt.plot(
    sample_route["cummulative_distance"],
    energy["cummulative_energy_kwh"],
    label="Tesla without Temperature",
)
plt.plot(
    sample_route["cummulative_distance"],
    energy_with_temp_72F["cummulative_energy_kwh"],
    label="Tesla with Temperature 72F",
)
plt.plot(
    sample_route["cummulative_distance"],
    energy_with_temp_32F["cummulative_energy_kwh"],
    label="Tesla with Temperature 32F",
)
plt.xlabel("Cumulative Distance (miles)")
plt.ylabel("Cumulative Energy (kWh)")
plt.legend()
# %%
"""
Notice how the energy consumption for the 32°F route is higher than the other two scenarios, reflecting the increased energy demand in colder temperatures.

Something else to note is that the model that doesn't consider temperature explicitly includes a "real world correction factor" to account for things like temperature _on average_.
This explains why the energy consumption for the 72°F route is slightly lower than the other two scenarios since the model without temperature adjustment is effectively averaging out the impact of temperature. 
The 72°F condition would be considered the "ideal" case since the vehicle does not have to expand any extra effort to maintain the cabin temperature.
"""
# %%
