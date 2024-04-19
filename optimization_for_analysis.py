# %% [markdown]
# # Black-formatter Example
#
# # Import libraries and data file

# %%
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import itertools

# xl = pd.ExcelFile("./SouthVietnam.xlsx")
xl = pd.ExcelFile("./SouthVietnam_change_firms_location.xlsx")
list_sheets = xl.sheet_names


population_df = pd.read_excel(
    xl, sheet_name="Population", engine="openpyxl", index_col=0
)

# supply_df = population_df.copy()
nb_of_periods = 6
growth_rate_a_year = 1 + 0.12
growth_rate_a_period = 1 + (growth_rate_a_year - 1) / 6
# # waste_per_person_per_month = 0.8
# # waste_per_person_per_year = waste_per_person_per_month * 12
# waste_per_person_per_year = 3.7 * 0.8
# sum_of_power = (growth_rate_a_period**nb_of_periods - 1) / (growth_rate_a_period - 1)
# supply_df["Supply of a year"] = supply_df["Populations"] * waste_per_person_per_year
#
# # Initialize periods
# for i in range(1, nb_of_periods + 1):
#     supply_df[i] = np.nan
# for i in supply_df.index:
#     supply_df[1].loc[i] = int(supply_df.loc[i]["Supply of a year"] / sum_of_power)
# for index in supply_df.index:
#     for i in range(2, nb_of_periods + 1):
#         supply_df[i].loc[index] = int(
#             supply_df[i - 1].loc[index] * growth_rate_a_period
#         )


# Share of products
product_shares = pd.read_excel(xl, sheet_name="Product_shares", index_col="Products")


def supply_of_product(periods, generation_points, products):
    return float(
        supply_df[periods].loc[generation_points]
        * product_shares["Share"].loc[products]
    )


# %%
establishing_cc_cost_df = pd.read_excel(xl, sheet_name="Establishing_cc", index_col=0)


def establishing_cc_cost(periods):
    return float(establishing_cc_cost_df.loc[periods]["Cost"])


# %% [markdown]
# # Distance from generation points to collection centers

# %%
# WARNING: Not test with Gurobi
distance_gp_cc_df = pd.read_excel(
    xl, sheet_name="Distance", engine="openpyxl", index_col=1
)
distance_gp_cc_df = distance_gp_cc_df.drop("No", axis=1)


def distance_gp_cc(generation_points, collection_centers):
    return float(distance_gp_cc_df[generation_points].loc[collection_centers])


# %% [markdown]
# # Distance from collection center to firms

# %%
# WARNING: Not test with Gurobi
distance_cc_firm_df = pd.read_excel(
    xl, sheet_name="DistancefromCCtoFirm", engine="openpyxl", index_col=0
)


def distance_cc_firm(collection_centers, firms):
    return float(distance_cc_firm_df[firms].loc[collection_centers])


# %% [markdown]
# # Unit transportation cost of truck to deliver products

# %%
transportation_cost_df = pd.read_excel(
    xl, sheet_name="Unit_transportation_cost", index_col=0
)


def transportation_cost_ij(periods):
    return float(transportation_cost_df["GP to CC"].loc[periods])


def transportation_cost_jf(periods):
    return float(transportation_cost_df["CC to Firm"].loc[periods])

def cost_of_buying_trucks(truck_types, products):
    return float(cost_of_buying_trucks_df[products].loc[truck_types])


# %% [markdown]
# # Stepwise quantity points

# %%
inventory_quantity_points_df = pd.read_excel(
    xl, sheet_name="Inventory_quantity_points", index_col=0
)
inventory_quantity_points = range(1, len(inventory_quantity_points_df) + 1)


def Q(inventory_quantity_points, products):
    return float(inventory_quantity_points_df[products].loc[inventory_quantity_points])


# %% [markdown]
# # Stepwise unit inventory cost

# %%
inventory_holding_costs_df = pd.read_excel(xl, sheet_name="Inventory_holding_costs")


def stepwise_inventory_unit_costs(periods, inventory_quantity_points, products):
    return float(
        inventory_holding_costs_df[
            (inventory_holding_costs_df["Period"] == periods)
            & (inventory_holding_costs_df["Step"] == inventory_quantity_points)
        ][products].item()
    )


# %%
fixed_costs_df = pd.read_excel(
    xl,
    # "./SouthVietnam_change_firms_location.xlsx",
    sheet_name="Location Index",
    # engine="openpyxl",
    index_col=0,
)
fixed_costs_df["Fixed_costs"] = np.nan
x = fixed_costs_df["Populations"].sum()
for i in fixed_costs_df.index:
    fixed_costs_df["Fixed_costs"][i] = 1 + fixed_costs_df["Populations"][i] / x

# %%
container_capacities_df = pd.read_excel(
    xl, sheet_name="ContainerCapacities", engine="openpyxl", index_col=0
)

cost_of_buying_trucks_df = pd.read_excel(xl, sheet_name= "Cost_of_buying_trucks",index_col=0)


def vehicle_cap(truck_types, products):
    return float(container_capacities_df[products].loc[truck_types])


#          ╭──────────────────────────────────────────────────────────╮
#          │                          Index                           │
#          ╰──────────────────────────────────────────────────────────╯

# %%
index_df = pd.read_excel(xl, sheet_name="Index")


def values_of_column(x):
    return index_df[x].dropna().unique()


products = values_of_column("Products")
generation_points = values_of_column("Generation points")
collection_centers = values_of_column("Collection centers")
firms = values_of_column("Firms")
periods = values_of_column("Periods")
periods = periods.astype(int)
truck_types = values_of_column("Truck types")


# %% [markdown]
# # Demand of firms

# %%
firms_share = pd.read_excel(xl, sheet_name="Firms_demand", index_col=0)


def demand_of_firms(firms, products):
    sum_of_supply_of_products = sum(
        supply_of_product(t, i, products) for t in periods for i in generation_points
    )
    return sum_of_supply_of_products * firms_share[firms].loc[products]


# for amount_each_year in np.arange(0.3, 3, 0.1):
for amount_each_year in np.arange(1, 1.1, 0.1):
    # for a in range(0, 20):
    for a in range(20, 20+1):
        supply_df = population_df.copy()
        # nb_of_periods = 6
        # growth_rate_a_year = 1 + 0.12
        # growth_rate_a_period = 1 + (growth_rate_a_year - 1) / 6
        # waste_per_person_per_month = 0.8
        # waste_per_person_per_year = waste_per_person_per_month * 12
        waste_per_person_per_year = 3.7 * amount_each_year * (a * 0.05)
        sum_of_power = (growth_rate_a_period**nb_of_periods - 1) / (
            growth_rate_a_period - 1
        )
        supply_df["Supply of a year"] = (
            supply_df["Populations"] * waste_per_person_per_year
        )

        # Initialize periods
        for i in range(1, nb_of_periods + 1):
            supply_df[i] = np.nan
        for i in supply_df.index:
            supply_df[1].loc[i] = int(
                supply_df.loc[i]["Supply of a year"] / sum_of_power
            )
        for index in supply_df.index:
            for i in range(2, nb_of_periods + 1):
                supply_df[i].loc[index] = int(
                    supply_df[i - 1].loc[index] * growth_rate_a_period
                )

        # %%
        model = gp.Model("")
        LC = 10000000
        SC = 0.0000001

        # Variable
        # model.setParam("MIPGap", 0.000)
        model.setParam("TimeLimit", 3600)

        x = model.addVars(
            periods,
            generation_points,
            collection_centers,
            products,
        )
        w = model.addVars(
            periods,
            collection_centers,
            firms,
            products,
        )
        inventory = model.addVars(
            periods,
            collection_centers,
            products,
        )
        # e = model.addVars(firms, products, lb=-GRB.INFINITY)
        e = model.addVars(firms, products)
        y = model.addVars(
            periods,
            collection_centers,
            vtype=GRB.BINARY,
        )
        z = model.addVars(
            periods,
            collection_centers,
            truck_types,
            products,
            vtype=GRB.BINARY,
        )
        u = model.addVars(
            periods,
            inventory_quantity_points,
            collection_centers,
            products,
            vtype=GRB.BINARY,
        )
        g = model.addVars(
            periods,
            inventory_quantity_points,
            collection_centers,
            products,
            # vtype=GRB.BINARY,
        )

        s = model.addVars(
            periods,
            generation_points,
            collection_centers,
            products,
            truck_types,
            vtype=GRB.INTEGER,
        )

        v = model.addVars(
            periods, collection_centers, firms, products, truck_types, vtype=GRB.INTEGER
        )

        m = model.addVars(
            periods,
            generation_points,
            collection_centers,
            products,
            truck_types,
            vtype=GRB.INTEGER,
        )

        n = model.addVars(
            periods, collection_centers, firms, products, truck_types, vtype=GRB.INTEGER
        )

        o = model.addVars(products, truck_types, vtype=GRB.INTEGER)
        q = model.addVars(products, truck_types, vtype=GRB.INTEGER)

        inventory_holding_unit_costs = model.addVars(
            periods,
            collection_centers,
            products,
        )

        S_pr = 15
        Q_pr = 15

        # OBJECTIVES
        TC_gp_cc = gp.quicksum(
            transportation_cost_ij(t) * distance_gp_cc(i, j) * s[t, i, j, p, k]
            for t in periods
            for i in generation_points
            for j in collection_centers
            for p in products
            for k in truck_types
        )
        TC_cc_firms = gp.quicksum(
            transportation_cost_jf(t) * distance_cc_firm(j, f) * v[t, j, f, p, k]
            for t in periods
            for j in collection_centers
            for f in firms
            for p in products
            for k in truck_types
        )
        # Costs_of_inventory = gp.quicksum(stepwise_inventory_unit_costs(t, a, p)*g[t,a,j,p] for t in periods for a in inventory_quantity_points for j in collection_centers for p in products)
        Costs_of_inventory = gp.quicksum(
            stepwise_inventory_unit_costs(t, 1, p) * g[t, 1, j, p]
            for t in periods
            for j in collection_centers
            for p in products
        )
        # Costs_of_establising_cc = gp.quicksum(establishing_cc_cost(t) * (y[t,j]-y[t-1,j]) for t in periods for j in collection_centers if t-1 in periods)
        Costs_of_establising_cc = gp.quicksum(
            establishing_cc_cost(t) * (y[t, j] - y[t - 1, j])
            for t in periods
            for j in collection_centers
            if t - 1 in periods
        ) + gp.quicksum(establishing_cc_cost(1) * y[1, j] for j in collection_centers)

        Costs_of_buying_trucks = gp.quicksum(cost_of_buying_trucks(r,p) *(o[p,r] + q[p,r]) for p in products for r in truck_types)

        model.update()

        model.setObjective(
            TC_gp_cc + TC_cc_firms + Costs_of_inventory + Costs_of_establising_cc + Costs_of_buying_trucks
        )

        # Constraint: (3)
        model.addConstrs(
            gp.quicksum(x[t, i, j, p] for j in collection_centers)
            == supply_of_product(t, i, p)
            for t in periods
            for i in generation_points
            for p in products
        )

        # Constraint: (4)
        model.addConstrs(
            gp.quicksum(x[t, i, j, p] for i in generation_points)
            + inventory[t - 1, j, p]
            == gp.quicksum(w[t, j, f, p] for f in firms) + inventory[t, j, p]
            for j in collection_centers
            for p in products
            for t in periods
            if t - 1 in periods
        )
        # model.addConstr(inventory[2, "Ho Chi Minh", "TV"] == 100)
        model.addConstrs(
            gp.quicksum(x[1, i, j, p] for i in generation_points)
            == gp.quicksum(w[1, j, f, p] for f in firms) + inventory[1, j, p]
            for j in collection_centers
            for p in products
        )

        # WARNING: Constraint: (5)
        model.addConstrs(
            gp.quicksum(
                w[t, j, f, p]
                for t in periods
                for j in collection_centers
                for t in periods
            )
            == demand_of_firms(f, p) + e[f, p]  # WARNING: D is parameter, not variable
            for f in firms
            for p in products
        )

        # Constraint: (6)

        # Constraint: (7)
        # model.addConstrs(gp.quicksum(x[t,i,j,p] for i in generation_points) <= y[t,j] for j in collection_centers for p in products for t in periods)

        # Constraint: (8)
        model.addConstrs(
            y[t - 1, j] <= y[t, j]
            for j in collection_centers
            for t in periods
            if t - 1 in periods
        )

        # Constraint: (9)
        # model.addConstrs(y[])

        # Constraint: (10)

        # Constraint: (11)

        # Constraint: (14)
        model.addConstrs(
            inventory_holding_unit_costs[t, j, p]
            == gp.quicksum(
                stepwise_inventory_unit_costs(t, a, p) * u[t, a, j, p]
                for a in inventory_quantity_points
                if a + 1 in inventory_quantity_points
            )
            for t in periods
            for j in collection_centers
            for p in products
        )

        # Constraint: (15)
        model.addConstrs(
            gp.quicksum(
                u[t, a, j, p] * Q(a, p)
                for a in inventory_quantity_points
                if a + 1 in inventory_quantity_points
            )
            <= inventory[t, j, p]
            for j in collection_centers
            for p in products
            for t in periods
        )
        model.addConstrs(
            inventory[t, j, p]
            <= gp.quicksum(
                u[t, a, j, p] * Q(a + 1, p)
                for a in inventory_quantity_points
                if a + 1 in inventory_quantity_points
            )
            for j in collection_centers
            for p in products
            for t in periods
        )

        # Constraint: (16)
        model.addConstrs(
            LC * (u[t, a, j, p] - 1) + inventory[t, j, p] <= g[t, a, j, p]
            for j in collection_centers
            for p in products
            for t in periods
            for a in inventory_quantity_points
            if a + 1 in inventory_quantity_points
        )
        model.addConstrs(
            g[t, a, j, p] <= LC * (1 - u[t, a, j, p]) + inventory[t, j, p]
            for j in collection_centers
            for p in products
            for t in periods
            for a in inventory_quantity_points
            if a + 1 in inventory_quantity_points
        )

        # Constraint: (17)
        model.addConstrs(
            -LC * u[t, a, j, p] <= g[t, a, j, p]
            for j in collection_centers
            for p in products
            for t in periods
            for a in inventory_quantity_points
            if a + 1 in inventory_quantity_points
        )
        model.addConstrs(
            g[t, a, j, p] <= LC * u[t, a, j, p]
            for j in collection_centers
            for p in products
            for t in periods
            for a in inventory_quantity_points
            if a + 1 in inventory_quantity_points
        )

        # Constraint: (18)
        model.addConstrs(
            gp.quicksum(
                u[t, a, j, p]
                for a in inventory_quantity_points
                if a + 1 in inventory_quantity_points
            )
            == 1
            for j in collection_centers
            for t in periods
            for p in products
        )

        # Constraint: (19)

        # Constraint: (20)
        model.addConstrs(
            gp.quicksum(
                vehicle_cap(k, p) * v[t, j, f, p, k]
                - vehicle_cap(k, p)
                + vehicle_cap(k, p) * SC
                for k in truck_types
            )
            <= w[t, j, f, p]
            for t in periods
            for j in collection_centers
            for f in firms
            for p in products
            # for k in truck_types
        )
        model.addConstrs(
            w[t, j, f, p]
            <= gp.quicksum(vehicle_cap(k, p) * v[t, j, f, p, k] for k in truck_types)
            for t in periods
            for j in collection_centers
            for f in firms
            for p in products
            # for k in truck_types
        )

        # Constraint: (21)
        model.addConstrs(
            gp.quicksum(
                vehicle_cap(k, p) * s[t, i, j, p, k]
                - vehicle_cap(k, p)
                + vehicle_cap(k, p) * SC
                for k in truck_types
            )
            <= x[t, i, j, p]
            for t in periods
            for j in collection_centers
            for i in generation_points
            for p in products
            # for k in truck_types
        )
        model.addConstrs(
            x[t, i, j, p]
            <= gp.quicksum(vehicle_cap(k, p) * s[t, i, j, p, k] for k in truck_types)
            for t in periods
            for j in collection_centers
            for i in generation_points
            for p in products
            # for k in truck_types
        )

        # Constraint: (22)
        model.addConstrs(
            -LC * y[t, j] <= x[t, i, j, p]
            for t in periods
            for i in generation_points
            for j in collection_centers
            for p in products
        )
        model.addConstrs(
            x[t, i, j, p] <= LC * y[t, j]
            for t in periods
            for i in generation_points
            for j in collection_centers
            for p in products
        )

        # Constraint: (21):
        model.addConstrs(
            S_pr * m[t, i, j, p, r] - S_pr + S_pr * SC <= s[t, i, j, p, r]
            for t in periods
            for i in generation_points
            for j in collection_centers
            for p in products
            for r in truck_types
        )
        model.addConstrs(
            s[t, i, j, p, r] <= S_pr * m[t, i, j, p, r]
            for t in periods
            for i in generation_points
            for j in collection_centers
            for p in products
            for r in truck_types
        )

        # Constraint: (22)
        model.addConstrs(
            Q_pr * n[t, j, f, p, r] - Q_pr + Q_pr * SC <= v[t, j, f, p, r]
            for t in periods
            for j in collection_centers
            for f in firms
            for p in products
            for r in truck_types
        )
        model.addConstrs(
            v[t, j, f, p, r] <= Q_pr * n[t, j, f, p, r]
            for t in periods
            for j in collection_centers
            for f in firms
            for p in products
            for r in truck_types
        )

        # Constraint: (23)
        model.addConstrs(
            o[p, r]
            >= gp.quicksum(
                m[t, i, j, p, r] for i in generation_points for j in collection_centers
            )
            for t in periods
            for p in products
            for r in truck_types
        )

        # Constraint: (24)
        model.addConstrs(
            q[p, r]
            >= gp.quicksum(n[t, j, f, p, r] for j in collection_centers for f in firms)
            for t in periods
            for p in products
            for r in truck_types
        )

        model.optimize()

        # %%
        file = open("Investigate_reverse_logistics.txt", mode="a")
        file.write(
            # "\n" + str(model.ObjVal)
            f"\n{amount_each_year*3.7} {a*0.05*100} {round(model.ObjVal,2)} {round(TC_gp_cc.getValue(),2)} {round(TC_cc_firms.getValue(),2)} {round(Costs_of_establising_cc.getValue(),2)} {round(Costs_of_inventory.getValue(),2)} {round(Costs_of_buying_trucks.getValue(), 2)} {model.MIPGap * 100} {round(model.Runtime,2)}"
        )
        file.close()
        model.dispose()
